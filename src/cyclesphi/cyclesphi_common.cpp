// #####################################################################################################################
// # Copyright(C) 2011-2025 IT4Innovations National Supercomputing Center, VSB - Technical University of Ostrava
// #
// # This program is free software : you can redistribute it and/or modify
// # it under the terms of the GNU General Public License as published by
// # the Free Software Foundation, either version 3 of the License, or
// # (at your option) any later version.
// #
// # This program is distributed in the hope that it will be useful,
// # but WITHOUT ANY WARRANTY; without even the implied warranty of
// # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
// # GNU General Public License for more details.
// #
// # You should have received a copy of the GNU General Public License
// # along with this program.  If not, see <https://www.gnu.org/licenses/>.
// #
// #####################################################################################################################

#include "cyclesphi_common.h"

#include <stdio.h>

#include "device/device.h"
#include "device/cuda/device_impl.h"

#include "scene/camera.h"
#include "scene/integrator.h"
#include "scene/scene.h"
#include "scene/background.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"

#include "scene/scene.h"
#include "scene/mesh.h"
#include "scene/geometry.h"
#include "scene/object.h"
#include "scene/light.h"

#include "util/vector.h"
#include "util/types.h"

#include "util/args.h"
#include "util/image.h"
#include "util/log.h"
#include "util/path.h"
#include "util/progress.h"
#include "util/string.h"
#include "util/time.h"
#include "util/transform.h"
#include "util/unique_ptr.h"
#include "util/version.h"
#include "util/vector.h"

//#include "cyclesphi/frame_output_driver.h"
#include "cyclesphi/frame_display_driver.h"
#include "renderengine_tcp.h"
#include "renderengine_data.h"
#include "cycles_xml_bin.h"

#include <omp.h>

//#define DEBUG_TIME

#ifdef DEBUG_TIME
#	define DEBUG_START_TIME(name) double t1_##name = omp_get_wtime();
#	define DEBUG_END_TIME(name) double t2_##name = omp_get_wtime(); printf("Elapsed time: %f, FPS: %f, %s\n", t2_##name - t1_##name, 1.0 / (t2_##name - t1_##name), #name);
#else
#	define DEBUG_START_TIME(name)
#	define DEBUG_END_TIME(name)
#endif

#ifdef WITH_CLIENT_GPUJPEG

#  ifdef WITH_CUDA_DYNLOAD
#    include "cuew.h"
#  else
#    include <cuda.h>
#  endif

/* Utility for checking return values of CUDA function calls. */
#ifndef cuda_assert

#  define cuda_assert(stmt) \
    { \
      CUresult result = stmt; \
      if (result != CUDA_SUCCESS) { \
        const char *name = cuewErrorString(result); \
        printf("%s in %s (%s:%d)", name, #stmt, __FILE__, __LINE__); \
      } \
    } \
    (void)0

#endif

#include "device/cuda/device.h"

#endif

//FromCL fromCL;

renderengine_data g_renderengine_data_rcv;
std::vector<renderengine_data> g_renderengine_datas;

double fps_previous_time = 0;
int fps_frame_count = 0;

static void session_print(const std::string& str)
{
	/* print with carriage return to overwrite previous */
	printf("\r%s", str.c_str());

	/* add spaces to overwrite longer previous print */
	static int maxlen = 0;
	int len = str.size();
	maxlen = std::max(len, maxlen);

	for (int i = len; i < maxlen; i++) {
		printf(" ");
	}

	/* flush because we don't write an end of line */
	fflush(stdout);
}

static void session_print_status(Options& options)
{
	std::string status, substatus;

	/* get status */
	double progress = options.session->progress.get_progress();
	options.session->progress.get_status(status, substatus);

	if (substatus != "") {
		status += ": " + substatus;
	}

	/* print status */
	status = ccl::string_printf("Progress %05.2f   %s", (double)progress * 100, status.c_str());
	session_print(status);
}

ccl::BufferParams& session_buffer_params(Options& options)
{
	static ccl::BufferParams buffer_params;
	buffer_params.width = options.width;
	buffer_params.height = options.height;
	buffer_params.full_width = options.width;
	buffer_params.full_height = options.height;

	return buffer_params;
}

void scene_init(Options& options)
{
	options.scene = options.session->scene.get();

	//const char* filepath_xml = getenv("CYCLES_XML_PATH");
	//if (filepath_xml == nullptr) {
	//	fprintf(stderr, "Missing CYCLES_XML_PATH\n");
	//	exit(-1);
	//}

	//std::string filepath_xml = options.filepath + std::string(".xml");

	xml_read_file(options.scene, options.filepath.c_str());

	/* Camera width/height override? */
	if (!(options.width == 0 || options.height == 0)) {
		options.scene->camera->set_full_width(options.width);
		options.scene->camera->set_full_height(options.height);
	}
	else {
		options.width = options.scene->camera->get_full_width();
		options.height = options.scene->camera->get_full_height();
	}

	/* Calculate Viewplane */
	options.scene->camera->compute_auto_viewplane();

	options.scene->integrator->set_use_adaptive_sampling(false);

	//options.scene->integrator->set_volume_step_rate(0.1f);
	//options.scene->integrator->set_volume_max_steps(64);
	//options.scene->integrator->set_max_volume_bounce(64);
}

void session_init(FromCL& fromCL, Options& options, int session_id)
{
	//const char* useGPU = getenv("CYCLESPHI_USE_GPU");
	std::string used_device = fromCL.used_device;
	options.filepath = fromCL.filepath;
	options.id = session_id;

	if (fromCL.use_anim && fromCL.anim > 1) {
		char temp[1024];
		sprintf(temp, "%05d", options.id);
		options.filepath = options.filepath + "_" + std::string(temp);
	}

	if (fromCL.use_mpi) {
		char temp[1024];
		sprintf(temp, "%05d", fromCL.world_rank);
		options.filepath = options.filepath + "_" + std::string(temp);
	}

	//std::string used_device = "CPU";
	//if (useGPU != nullptr) {
	//	used_device = useGPU;
	//}

	//options.session_params.device.type = ccl::DEVICE_CPU;
	//if (useGPU) 
	printf("used Device: %s\n", used_device.c_str());

	/* find matching device */
	ccl::DeviceType device_type = ccl::Device::type_from_string(used_device.c_str());
	ccl::vector<ccl::DeviceInfo> devices = ccl::Device::available_devices((ccl::DeviceTypeMask)(1 << device_type));

	printf("Devices:\n");

	for (const ccl::DeviceInfo& info : devices) {
		printf("    %-10s%s%s\n",
			ccl::Device::string_from_type(info.type).c_str(),
			info.description.c_str(),
			(info.display_device) ? " (display)" : "");

	}

	bool device_available = false;
	if (!devices.empty()) {
		options.session_params.device = ccl::Device::get_multi_device(devices, 0, false); //devices.front();
		device_available = true;
	}

	/* handle invalid configurations */
	if (options.session_params.device.type == ccl::DEVICE_NONE || !device_available) {
		fprintf(stderr, "Unknown device: %s\n", used_device.c_str());
		exit(-1);
	}

	options.session_samples = 0;

	options.session_params.background = false;
	options.session_params.headless = false;
	options.session_params.use_auto_tile = false;
	options.session_params.tile_size = 16;
	options.session_params.use_resolution_divider = false;
	options.session_params.samples = 1;

	//options.session_params.threads = 1;

	options.output_pass = "combined";

	options.session = new ccl::Session(options.session_params, options.scene_params);

	// if (!options.output_filepath.empty()) {
	//   options.session->set_output_driver(make_unique<OIIOOutputDriver>(
	//       options.output_filepath, options.output_pass, session_print));
	// }

//#ifdef WITH_CLIENT_GPUJPEG
//#if 1//def WITH_CLIENT_GPUJPEG
//	if (options.session_params.device.type != ccl::DEVICE_CPU) {
	auto display_driver = std::make_unique<ccl::FrameDisplayDriver>();
	options.display_driver = display_driver.get();
	options.session->set_display_driver(std::move(display_driver));
	//	}
	//	else 
	//#endif
	//	{
	//		//#else
	//		auto output_driver = std::make_unique<ccl::FrameOutputDriver>(options.output_pass, session_print);
	//		options.output_driver = output_driver.get();
	//		options.session->set_output_driver(std::move(output_driver));
	//		//#endif
	//	}

	if (options.session_params.background && !options.quiet) {
		options.session->progress.set_update_callback([&options]() {session_print_status(options); });
	}

	/* load scene */
	scene_init(options);

	/* add pass for output. */
	ccl::Pass* pass = options.scene->create_node<ccl::Pass>();
	pass->set_name(ccl::ustring(options.output_pass.c_str()));
	pass->set_type(ccl::PASS_COMBINED);

	///////////////////////////////////////////
	//auto* shader = options.scene->default_background;
	//auto* graph = new ccl::ShaderGraph();

	//// Create the background node
	//auto* bg = graph->create_node<ccl::BackgroundNode>();
	//bg->set_strength(1.0f);
	//bg->set_color(ccl::make_float3(0.3f, 0.3f, 0.3f));
	//bg->set_owner(graph);
	//graph->add(bg);

	//graph->connect(bg->output("Background"), graph->output()->input("Surface"));

	//shader->set_graph(graph);
	//options.scene->background->set_shader(options.scene->default_background);
	//options.scene->background->set_use_shader(true);
	//shader->tag_update(options.scene);
	///////////////////////////////////////////

	//options.session->reset(options.session_params, session_buffer_params());
	//options.session->start();
}

void session_exit(FromCL& fromCL, Options& options)
{
	if (options.session) {
		delete options.session;
		options.session = NULL;
	}

	if (options.session_params.background && !options.quiet) {
		session_print("Finished Rendering.");
		printf("\n");
	}
}

void renderFrame(Options* options)
{
	if (options->display_driver)
		options->display_driver->renderBegin();

	if (options->session_samples == 0) { // reset
		options->session->reset(options->session_params, session_buffer_params(*options));
	}

	//if(options->output_driver)
	//	options->output_driver->renderBegin();

	options->session->set_samples(++options->session_samples);
	options->session->start();

	//options->session->wait();
	options->session->draw();

	//if (options->output_driver)
	//	options->output_driver->wait();		

	if (options->display_driver)
		options->display_driver->wait();
}

struct CyclesphiDataRenderAux {
	std::vector<char> data;
};

///////////////////////////////////////RENDERING//////////////////////////////////////////////////////
int cyclesphi(int ac, char** av, TcpConnection* blenderClientTcp, FromCL& fromCL, std::vector<Options>& options)
{
	////////////////////////////////////////////////////
	double render_time = 0;
	double render_time_accu = 0;
	int spp_one_step = 0;

#if 0//def WITH_CLIENT_GPUJPEG
	CUdeviceptr cuda_fb = NULL;
	CUdevice cuDevice;
	CUcontext cuContext;

	ccl::device_cuda_init();

	//cuewInit(CUEW_INIT_CUDA);
	//cuda_assert(cuInit(0));
	cuda_assert(cuDeviceGet(&cuDevice, 0));
	cuda_assert(cuCtxCreate(&cuContext, CU_CTX_LMEM_RESIZE_TO_MAX, cuDevice));
#endif

	//#ifdef WITH_CLIENT_GPUJPEG
	//	void* cuda_fb = NULL;
	//#endif

	std::vector<char> pixels_buf_empty;
	CyclesphiDataRenderAux data_render_aux_rcv;
	std::vector<CyclesphiDataRenderAux> data_render_aux;

	//TransferFunction xf;
	//xf.colorMap.resize(sizeof(cyclesphiDataRender.colorMap) / sizeof(vec4f));
	//xf.domain = range1f(0.f, 1.0f);

	//int total_samples = 0;
	//cyclesphi::Camera camera;

	//double t2 = getCurrentTime();

	/////////
	g_renderengine_datas.resize(options.size());

	Options* main_options = &options[0];
	renderengine_data* main_renderengine_data = &g_renderengine_datas[0];

	data_render_aux.resize(options.size());
	CyclesphiDataRenderAux* main_data_render_aux = &data_render_aux[0];
	/////////

	BRaaSHPCDataState cyclesphiDataState;
	memset(&cyclesphiDataState, 0, sizeof(cyclesphiDataState));

	///////////////////
	ccl::BoundBox bbox_scene = ccl::BoundBox::empty;
	bool bbox_computed = false;
	///////////////////
//
//#ifdef WITH_CLIENT_GPUJPEG
//	ccl::device_vector<ccl::half4> temp_buffer(options.session->device, "buffer", ccl::MemoryType::MEM_DEVICE_ONLY);
//#endif

	//cyclesphiDataState.world_bounds_spatial_lower[0] = worldBounds.spatial.lower[0];
	//cyclesphiDataState.world_bounds_spatial_lower[1] = worldBounds.spatial.lower[1];
	//cyclesphiDataState.world_bounds_spatial_lower[2] = worldBounds.spatial.lower[2];
	//cyclesphiDataState.world_bounds_spatial_upper[0] = worldBounds.spatial.upper[0];
	//cyclesphiDataState.world_bounds_spatial_upper[1] = worldBounds.spatial.upper[1];
	//cyclesphiDataState.world_bounds_spatial_upper[2] = worldBounds.spatial.upper[2];

	//cyclesphiDataState.scalars_range[0] = worldBounds.scalars.lo;
	//cyclesphiDataState.scalars_range[1] = worldBounds.scalars.hi;

	/////////
	fromCL.render_running = true;

	session_print("Start rendering...\n");

	while (fromCL.render_running) {
		DEBUG_START_TIME(overall);

		DEBUG_START_TIME(receive);

		blenderClientTcp->recv_data_data((char*)&g_renderengine_data_rcv, sizeof(renderengine_data));
		if (blenderClientTcp->is_error()) {
			//throw std::runtime_error("TCP Error!");
			break;
		}

		if (g_renderengine_data_rcv.reset) {
			// if (renderer != NULL) {
			// 	delete renderer;
			// 	renderer = NULL;
			// }

			//blenderClientTcp->client_close();
			//blenderClientTcp->server_close();

			break;
		}

		if (g_renderengine_data_rcv.width == 0 || g_renderengine_data_rcv.height == 0) {
			printf("width or height is 0!!!!\n");
			fflush(0);
			//exit(-1);
			break;
		}

		blenderClientTcp->set_frame(g_renderengine_data_rcv.frame);

		// check animation
		if (options.size() > 1 && blenderClientTcp->get_frame() >= 0 && blenderClientTcp->get_frame() < options.size()) {
			main_options = &options[blenderClientTcp->get_frame()];
			main_renderengine_data = &g_renderengine_datas[blenderClientTcp->get_frame()];
			main_data_render_aux = &data_render_aux[blenderClientTcp->get_frame()];
		}

		g_renderengine_data_rcv.frame = main_renderengine_data->frame;

		int cyclesphiDataRenderSize = 0;
		blenderClientTcp->recv_data_data((char*)&cyclesphiDataRenderSize, sizeof(int));

		if (data_render_aux_rcv.data.size() != cyclesphiDataRenderSize) {
			data_render_aux_rcv.data.resize(cyclesphiDataRenderSize);
		}

		if (data_render_aux_rcv.data.size() > 0) {
			blenderClientTcp->recv_data_data((char*)data_render_aux_rcv.data.data(), data_render_aux_rcv.data.size());
			data_render_aux_rcv.data.push_back('\0');
		}

		if (blenderClientTcp->is_error()) {
			//throw std::runtime_error("TCP Error!");
			break;
		}

		if (pixels_buf_empty.size() != sizeof(ccl::half4) * g_renderengine_data_rcv.width * g_renderengine_data_rcv.height) {
			pixels_buf_empty.resize(sizeof(ccl::half4) * g_renderengine_data_rcv.width * g_renderengine_data_rcv.height);
		}

		DEBUG_END_TIME(receive);

		try {
			// cam_change
			if (/*renderer == NULL || */ memcmp(main_renderengine_data, &g_renderengine_data_rcv, sizeof(renderengine_data))) {
				DEBUG_START_TIME(camera);
				memcpy(main_renderengine_data, &g_renderengine_data_rcv, sizeof(renderengine_data));

				//Camera cam = renderer->getCamera();
				//renderer->resetAccumulation();				
				//total_samples = 0;
				render_time = 0;
				render_time_accu = 0;
				//renderer->config.camera.dirty = true;

				main_options->session_samples = 0;

				if (g_renderengine_data_rcv.reset || main_options->width != g_renderengine_data_rcv.width
					|| main_options->height != g_renderengine_data_rcv.height) {

					main_options->width = g_renderengine_data_rcv.width;
					main_options->height = g_renderengine_data_rcv.height;

#if 0 //def WITH_CLIENT_GPUJPEG
					//cuDeviceSet(0);
					cuda_assert(cuCtxPushCurrent(cuContext));

					if (cuda_fb)
						cuda_assert(cuMemFree(cuda_fb));

					//cuda_assert(cuMemAlloc(&cuda_fb, main_options->width * main_options->height * sizeof(ccl::half4)));
					cuda_assert(cuMemAllocManaged(&cuda_fb, main_options->width * main_options->height * sizeof(ccl::half4), CU_MEM_ATTACH_GLOBAL));

					cuda_assert(cuCtxPopCurrent(NULL));

					main_options->output_driver->d_pixels = (void*)cuda_fb;
#endif

#if 0 //def WITH_CLIENT_GPUJPEG
					cuda_assert(cudaSetDevice(0)); // Set the appropriate device if not already set

					if (cuda_fb)
						cuda_assert(cudaFree(cuda_fb));

					cuda_assert(cudaMalloc(&cuda_fb, main_options->width * main_options->height * sizeof(uint32_t)));
					// cudaMallocManaged(&cuda_fb, main_options->width * main_options->height * sizeof(uint32_t), cudaMemAttachGlobal);

					main_options->output_driver->d_pixels = (void*)cuda_fb;
#endif

					//#ifdef WITH_CLIENT_GPUJPEG
					//					temp_buffer.alloc(main_options->width, main_options->height);
					//					temp_buffer.zero_to_device();
					//					if (main_options->output_driver)
					//						main_options->output_driver->d_pixels = (void*)temp_buffer.device_pointer;
					//
					//					//if (main_options->display_driver)
					//					//	main_options->display_driver->d_pixels = (void*)temp_buffer.device_pointer;
					//#endif

										//renderer->resize(fromCL.fbSize, (uint32_t*)cuda_fb);
				}

				float* input = g_renderengine_data_rcv.cam.transform_inverse_view_matrix;

				// Convert to ccl::Transform
				ccl::Transform tfm = ccl::transform_clear_scale(ccl::make_transform(
					input[0], input[1], input[2],   // First row
					input[3], input[4], input[5],   // Second row
					input[6], input[7], input[8],   // Third row
					input[9], input[10], input[11]  // Fourth row (Translation vector)
				) * ccl::transform_scale(1.0f, 1.0f, -1.0f));

				main_options->scene->camera->set_matrix(tfm);
				main_options->scene->camera->set_full_width(main_options->width);
				main_options->scene->camera->set_full_height(main_options->height);

				main_options->scene->camera->set_nearclip(g_renderengine_data_rcv.cam.clip_start);
				main_options->scene->camera->set_farclip(g_renderengine_data_rcv.cam.clip_end);

				main_options->scene->camera->need_flags_update = true;
				main_options->scene->camera->need_device_update = true;

				//perspective
				main_options->scene->camera->set_fov(g_renderengine_data_rcv.cam.lens);

				if (g_renderengine_data_rcv.cam.view_perspective == 1) { //CAMERA_ORTHOGRAPHIC
					main_options->scene->camera->set_camera_type(ccl::CameraType::CAMERA_ORTHOGRAPHIC);
				}
				else {
					main_options->scene->camera->set_camera_type(ccl::CameraType::CAMERA_PERSPECTIVE);
				}

				float xratio = (float)main_options->width;
				float yratio = (float)main_options->height;
				bool horizontal_fit = (xratio > yratio);

				float aspectratio;
				float xaspect, yaspect;
				if (horizontal_fit) {
					aspectratio = xratio / yratio;
					xaspect = aspectratio;
					yaspect = 1.0f;
				}
				else {
					aspectratio = yratio / xratio;
					xaspect = 1.0f;
					yaspect = aspectratio;
				}

				if (g_renderengine_data_rcv.cam.view_perspective == 1) { //CAMERA_ORTHOGRAPHIC
					float ortho_scale = g_renderengine_data_rcv.cam.lens / 2.0f;
					xaspect = xaspect * ortho_scale;// / (aspectratio * 2.0f);
					yaspect = yaspect * ortho_scale;// / (aspectratio * 2.0f);
					//aspectratio = ortho_scale / 2.0f;					
				}

				main_options->scene->camera->set_viewplane_left(-xaspect);
				main_options->scene->camera->set_viewplane_right(xaspect);
				main_options->scene->camera->set_viewplane_bottom(-yaspect);
				main_options->scene->camera->set_viewplane_top(yaspect);

				DEBUG_END_TIME(camera);
			}

			// if (renderer == NULL) {
			// 	continue;
			// }

			if (data_render_aux_rcv.data.size() != main_data_render_aux->data.size()) {
				DEBUG_START_TIME(resize_rcv_data);
				main_data_render_aux->data.resize(data_render_aux_rcv.data.size());
				DEBUG_END_TIME(resize_rcv_data);
			}

			if (data_render_aux_rcv.data.size() > 0 && memcmp(main_data_render_aux->data.data(), data_render_aux_rcv.data.data(), data_render_aux_rcv.data.size())) {
				DEBUG_START_TIME(material);
				memcpy(main_data_render_aux->data.data(), data_render_aux_rcv.data.data(), data_render_aux_rcv.data.size());

				//memcpy(xf.colorMap.data(), cyclesphiDataRender.colorMap, sizeof(vec4f) * xf.colorMap.size());
				//xf.domain = range1f(cyclesphiDataRender.domain[0], cyclesphiDataRender.domain[1]);
				//xf.baseDensity = cyclesphiDataRender.baseDensity;

				//Camera cam = renderer->getCamera();
				//renderer->resetAccumulation();
				main_options->session_samples = 0;
				//total_samples = 0;
				render_time = 0;
				render_time_accu = 0;
				//renderer->config.camera.dirty = true;

				////renderer->setColorMap(haystack_data);
				//renderer->setTransferFunction(xf);
				//renderer->resetAccumulation();
				//total_samples = 0;

				//const char* mat_name = getenv("CYCLES_MAT_NAME");
				//const char* node_name = getenv("CYCLES_MAT_NODE");

				//if (mat_name && node_name) 
				{
					//xml_set_volume_to_attr(main_options->scene, volume_geom_name, volume_attr_name, file_type, grid_handle_final);
					//xml_set_material_to_node(main_options->scene, cyclesphiDataRender.data());
					xml_set_material_to_shader(main_options->scene, main_data_render_aux->data.data());
				}
				DEBUG_END_TIME(material);
			}

			/////////////////////////////////////////////////
			DEBUG_START_TIME(render);
			renderFrame(main_options);
			DEBUG_END_TIME(render);

#ifdef WITH_CLIENT_GPUJPEG     
			if (main_options->display_driver) {
				DEBUG_START_TIME(send_gpujpeg_display);
				blenderClientTcp->send_gpujpeg((char*)main_options->display_driver->d_pixels, pixels_buf_empty.data(), main_options->width, main_options->height, 8);
				//blenderClientTcp->send_gpujpeg((char*)main_options->display_driver->pixels.data(), pixels_buf_empty.data(), main_options->width, main_options->height, 0);
				DEBUG_END_TIME(send_gpujpeg_display);
			}
			//else if (main_options->output_driver) {
			//	DEBUG_START_TIME(send_gpujpeg_output);
			//	blenderClientTcp->send_gpujpeg((char*)main_options->output_driver->pixels.data(), pixels_buf_empty.data(), main_options->width, main_options->height, 1);
			//	DEBUG_END_TIME(send_gpujpeg_output);
			//}
#else
			//char* pixels_buf = (char*)main_options->output_driver->pixels.data(); //cuda_fb; //renderer->getBuffer();
			//((int*)pixels_buf)[0] = total_samples; //renderer->getTotalSamples();

			//blenderClientTcp->save_bmp(main_options->width, main_options->height, (char*)main_options->output_driver->pixels.data(), main_options->session_samples);
			//DEBUG_START_TIME(send_output);
			//blenderClientTcp->send_data_data((char*)main_options->output_driver->pixels.data(), pixels_buf_empty.size());
			//DEBUG_END_TIME(send_output);

			if (main_options->display_driver) {
				DEBUG_START_TIME(send_gpujpeg_display);
				blenderClientTcp->send_data_data((char*)main_options->display_driver->pixels.data(), pixels_buf_empty.size());
				DEBUG_END_TIME(send_gpujpeg_display);
			}
			//else if (main_options->output_driver) {
			//	DEBUG_START_TIME(send_gpujpeg_output);
			//	blenderClientTcp->send_data_data((char*)main_options->output_driver->pixels.data(), pixels_buf_empty.size());
			//	DEBUG_END_TIME(send_gpujpeg_output);
			//}
#endif
			if (blenderClientTcp->is_error()) {
				throw std::runtime_error("TCP Error!");
			}

			if (!bbox_computed) {
				DEBUG_START_TIME(bbox_computed);
				for (ccl::Object* object : main_options->scene->objects) {
					bbox_scene.grow(object->bounds);
				}

				cyclesphiDataState.world_bounds_spatial_lower[0] = bbox_scene.min[0];
				cyclesphiDataState.world_bounds_spatial_lower[1] = bbox_scene.min[1];
				cyclesphiDataState.world_bounds_spatial_lower[2] = bbox_scene.min[2];
				cyclesphiDataState.world_bounds_spatial_upper[0] = bbox_scene.max[0];
				cyclesphiDataState.world_bounds_spatial_upper[1] = bbox_scene.max[1];
				cyclesphiDataState.world_bounds_spatial_upper[2] = bbox_scene.max[2];

				bbox_computed = true;
				DEBUG_END_TIME(bbox_computed);
			}

			DEBUG_START_TIME(send_data_state);
			float duration = 0;
			//if (main_options->output_driver)
			//	duration = main_options->output_driver->duration;
			if (main_options->display_driver)
				duration = main_options->display_driver->duration;

			cyclesphiDataState.fps = (float)main_options->session_params.samples / duration;//fps;
			cyclesphiDataState.samples = main_options->session_samples;//total_samples;
			blenderClientTcp->send_data_data((char*)&cyclesphiDataState, sizeof(cyclesphiDataState));
			DEBUG_END_TIME(send_data_state);

			if (blenderClientTcp->is_error()) {
				throw std::runtime_error("TCP Error!");
			}
		}
		catch (const std::exception& ex)
		{
			std::cerr << ex.what();
			//exit(-1);
			break;
		}

		DEBUG_END_TIME(overall);
	}

	//blenderClientTcp->client_close();
	//blenderClientTcp->server_close();

	//////////////////////////////////////////////////// 	

	// reset
	memset(&g_renderengine_data_rcv, 0, sizeof(g_renderengine_data_rcv));

	g_renderengine_datas.clear();
	//memset(&main_renderengine_data, 0, sizeof(main_renderengine_data));
	//memset(&fromCL.fbSize, 0, sizeof(fromCL.fbSize));

	//if (fbPointer)
	//	cudaFree(fbPointer);	

	return 0;
}


/////////////////////
void FromCL::usage()
{
	std::cout << "./cyclesphi <options>" << std::endl;

	std::cout << "options:" << std::endl;
	std::cout << "\t--scene X" << std::endl;
	std::cout << "\t--device X" << std::endl;
	std::cout << "\t--port X" << std::endl;
	std::cout << "\t--anim X" << std::endl;

	const ccl::vector<ccl::DeviceInfo> devices = ccl::Device::available_devices();
	printf("Devices:\n");

	for (const ccl::DeviceInfo& info : devices) {
		printf("    %-10s%s%s\n",
			ccl::Device::string_from_type(info.type).c_str(),
			info.description.c_str(),
			(info.display_device) ? " (display)" : "");
	}

	exit(0);
}

void FromCL::parse_args(int argc, char** argv)
{
	//if (argc < 2) {
	//	usage();
	//}

	for (int i = 1; i < argc; i++) {
		const std::string arg = argv[i];
		if (arg == "--port") {
			port = std::stoi(argv[++i]);
		}
		else if (arg == "--anim") {
			use_anim = true;
			anim = std::stoi(argv[++i]);
		}
		else if (arg == "--scene") {
			filepath = argv[++i];
		}
		else if (arg == "--device") {
			used_device = argv[++i];
		}
		else if (arg == "-h" || arg == "--help") {
			usage();
		}
	}
}