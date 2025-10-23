/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cyclesphi/frame_output_driver.h"

#include "scene/colorspace.h"

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

CCL_NAMESPACE_BEGIN

FrameOutputDriver::FrameOutputDriver(const string_view pass,
	LogFunction log)
	: pass_(pass), log_(log), render_finished(true), duration(0.0f), d_pixels(nullptr)
{
}

FrameOutputDriver::~FrameOutputDriver()
{
}

void FrameOutputDriver::renderBegin()
{
	//std::lock_guard<std::mutex> lock(mutex);
	thread_scoped_lock lock(mutex);
	start = std::chrono::steady_clock::now();
	render_finished = false;
	//render_finished.store(false, std::memory_order_release);
	//print("renderBegin()\n"); fflush(0);
}

void FrameOutputDriver::renderEnd()
{
	//std::lock_guard<std::mutex> lock(mutex);
	thread_scoped_lock lock(mutex);

	auto end = std::chrono::steady_clock::now();
	duration = std::chrono::duration<float>(end - start).count();

	render_finished = true;
	//render_finished.store(true, std::memory_order_release);

	// Notify the wait thread
	cv.notify_all();
	//print("renderEnd()\n"); fflush(0);

#if 0 //def _WIN32
	std::this_thread::sleep_for(std::chrono::milliseconds(5)); //TODO
#endif
}

void FrameOutputDriver::wait()
{
	//thread_scoped_lock lock(mutex);
	//cv.wait(lock, [this] { return render_finished; });

	while (true) {
		thread_scoped_lock session_thread_lock(mutex);
		if (render_finished) {
			break;
		}
		cv.wait(session_thread_lock);
	}
}

bool FrameOutputDriver::ready() const
{
	return render_finished;
	//return render_finished.load(std::memory_order_acquire);
}

void FrameOutputDriver::write_render_tile(const Tile& tile)
{
	/* Only write the full buffer, no intermediate tiles. */
	if (!(tile.size == tile.full_size)) {
		renderEnd();
		return;
	}

	const int width = tile.size.x;
	const int height = tile.size.y;

	if (d_pixels == nullptr) {
		if (pixels.size() != width * height) {
			pixels.resize(width * height);
		}

		if (!tile.get_pass_pixels(pass_, false, sizeof(half4), pixels.data())) {
			log_("Failed to read render pass pixels");
			renderEnd();
			return;
		}
	}
	else {
		if (!tile.get_pass_pixels(pass_, true, sizeof(half4), d_pixels)) {
			log_("Failed to read render pass pixels");
			renderEnd();
			return;
		}
	}

	renderEnd();
}

CCL_NAMESPACE_END
