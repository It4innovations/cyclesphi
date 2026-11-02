"""Cycles XML test scene of a .srnxr version 2 neural volume.

  python srnxr_make_scene.py model.srnxr out.xml [--time 0.5] [--width 960 --height 540]
                             [--density-scale 8] [--emission-scale 0.6] [--step 0.0]

Writes out.xml and the (empty) out.xml.bin the XML loader expects. The volume material is the
transfer function, outside of the decoder:
  density  = max(rho, 0) * density_scale                     (attribute "neural_density", field rho)
  emission = max(p, 0) * emission_scale, orange              (attribute "pressure", field p)
The backend is chosen on the command line (cycles --neural-backend ...). Pass the resolution to
cycles as well (--width/--height): the standalone ignores the camera resolution of the XML. The attributes are
named attributes (ATTR_STD_NONE): an attribute node reading "density" did not find a voxel
attribute declared with std="ATTR_STD_VOLUME_DENSITY" in the XML loader.
"""

from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("output")
    ap.add_argument("--time", type=float, default=0.5)
    ap.add_argument("--time-shutter", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--density-scale", type=float, default=8.0)
    ap.add_argument("--emission-scale", type=float, default=0.6)
    ap.add_argument("--density-field", default="rho")
    ap.add_argument("--emission-field", default="p")
    ap.add_argument("--no-emission", action="store_true")
    ap.add_argument("--step", type=float, default=0.0,
                    help="volume step size, > 0 for ray marching scenes")
    args = ap.parse_args()

    model = os.path.abspath(args.model).replace("\\", "/")
    t = args.time
    emission_attr = "" if args.no_emission else f'''
		<attribute name="pressure" std="ATTR_STD_NONE" type="float" element="ATTR_ELEMENT_VOXEL" flags="0" volume_type="srnxr" buffer="{model}" srnxr_fields="{args.emission_field}" srnxr_time="{t}" srnxr_time_shutter="{args.time_shutter}" />'''
    emission_nodes = "" if args.no_emission else f'''
		<attribute name="attr_pressure" ui_name="Attribute">
			<socket name="attribute" ui_name="Attribute" value="pressure" type_name="string" />
		</attribute>
		<math name="pressure_clamp" ui_name="Math">
			<socket name="math_type" ui_name="Type" value="maximum" type_name="enum" />
			<socket name="value2" ui_name="Value2" value="0" type_name="float" />
		</math>
		<math name="pressure_scale" ui_name="Math">
			<socket name="math_type" ui_name="Type" value="multiply" type_name="enum" />
			<socket name="value2" ui_name="Value2" value="{args.emission_scale}" type_name="float" />
		</math>
		<connect from_node="attr_pressure" from_socket="fac" to_node="pressure_clamp" to_socket="value1" />
		<connect from_node="pressure_clamp" from_socket="value" to_node="pressure_scale" to_socket="value1" />
		<connect from_node="pressure_scale" from_socket="value" to_node="volume_node" to_socket="emission_strength" />'''
    step = f'\n\t\t<socket name="step_size" ui_name="Step Size" value="{args.step}" type_name="float" />' \
        if args.step > 0 else ""
    xml = f'''<?xml version="1.0"?>
<cycles>
	<shader name="default_surface" ui_name="default_surface">
		<principled_bsdf name="surface_bsdf" ui_name="principled_bsdf" />
		<connect from_node="surface_bsdf" from_socket="BSDF" to_node="output" to_socket="surface" />
	</shader>
	<shader name="default_volume" ui_name="default_volume" />
	<shader name="default_light" ui_name="default_light" />
	<shader name="default_background" ui_name="default_background">
		<background_shader name="world_node" ui_name="Background">
			<socket name="color" ui_name="Color" value="0.05 0.05 0.06 0" type_name="color" />
			<socket name="strength" ui_name="Strength" value="1" type_name="float" />
		</background_shader>
		<connect from_node="world_node" from_socket="background" to_node="output" to_socket="surface" />
	</shader>
	<shader name="default_empty" ui_name="default_empty" />
	<shader name="light_shader" ui_name="Light">
		<emission name="light_emission" ui_name="Emission">
			<socket name="color" ui_name="Color" value="1 1 1 0" type_name="color" />
			<socket name="strength" ui_name="Strength" value="1" type_name="float" />
		</emission>
		<connect from_node="light_emission" from_socket="emission" to_node="output" to_socket="surface" />
	</shader>
	<shader name="volume_material" ui_name="NeuralVolumeMaterial">
		<attribute name="attr_density" ui_name="Attribute">
			<socket name="attribute" ui_name="Attribute" value="neural_density" type_name="string" />
		</attribute>
		<math name="density_clamp" ui_name="Math">
			<socket name="math_type" ui_name="Type" value="maximum" type_name="enum" />
			<socket name="value2" ui_name="Value2" value="0" type_name="float" />
		</math>
		<math name="density_scale" ui_name="Math">
			<socket name="math_type" ui_name="Type" value="multiply" type_name="enum" />
			<socket name="value2" ui_name="Value2" value="{args.density_scale}" type_name="float" />
		</math>
		<principled_volume name="volume_node" ui_name="Principled Volume">
			<socket name="color" ui_name="Color" value="0.85 0.85 0.9 0" type_name="color" />
			<socket name="emission_color" ui_name="Emission Color" value="1.0 0.45 0.1 0" type_name="color" />
		</principled_volume>
		<connect from_node="attr_density" from_socket="fac" to_node="density_clamp" to_socket="value1" />
		<connect from_node="density_clamp" from_socket="value" to_node="density_scale" to_socket="value1" />
		<connect from_node="density_scale" from_socket="value" to_node="volume_node" to_socket="density" />{emission_nodes}
		<connect from_node="volume_node" from_socket="volume" to_node="output" to_socket="volume" />
	</shader>
	<camera name="camera" ui_name="camera">
		<socket name="matrix" ui_name="Matrix" value="0.680451 -0.321659 -0.658424 4.2 0.732793 0.298683 0.611393 -3.9 -0 0.898512 -0.438949 2.8" type_name="transform" />
		<socket name="fov" ui_name="FOV" value="0.45" type_name="float" />
		<socket name="nearclip" ui_name="Near Clip" value="0.1" type_name="float" />
		<socket name="farclip" ui_name="Far Clip" value="100" type_name="float" />
		<socket name="viewplane.left" ui_name="Viewplane Left" value="{-args.width / args.height:.5f}" type_name="float" />
		<socket name="viewplane.right" ui_name="Viewplane Right" value="{args.width / args.height:.5f}" type_name="float" />
		<socket name="viewplane.bottom" ui_name="Viewplane Bottom" value="-1" type_name="float" />
		<socket name="viewplane.top" ui_name="Viewplane Top" value="1" type_name="float" />
		<socket name="full_width" ui_name="Full Width" value="{args.width}" type_name="int" />
		<socket name="full_height" ui_name="Full Height" value="{args.height}" type_name="int" />
	</camera>
	<film name="film" ui_name="film">
		<socket name="filter_type" ui_name="Filter Type" value="blackman_harris" type_name="enum" />
		<socket name="filter_width" ui_name="Filter Width" value="1.5" type_name="float" />
	</film>
	<background name="background" ui_name="background" />
	<integrator name="integrator" ui_name="integrator">
		<socket name="max_bounce" ui_name="Max Bounce" value="4" type_name="int" />
		<socket name="max_volume_bounce" ui_name="Max Volume Bounce" value="1" type_name="int" />
		<socket name="use_adaptive_sampling" ui_name="Use Adaptive Sampling" value="false" type_name="boolean" />
		<socket name="seed" ui_name="Seed" value="0" type_name="int" />
	</integrator>
	<volume name="neural_volume" ui_name="NeuralVolume" geometry_type="2">
		<socket name="used_shaders" ui_name="Shaders" value="volume_material" type_name="array_node" />{step}
		<attribute name="neural_density" std="ATTR_STD_NONE" type="float" element="ATTR_ELEMENT_VOXEL" flags="0" volume_type="srnxr" buffer="{model}" srnxr_fields="{args.density_field}" srnxr_time="{t}" srnxr_time_shutter="{args.time_shutter}" />{emission_attr}
	</volume>
	<pointlight name="key_light" ui_name="Light" geometry_type="6">
		<socket name="used_shaders" ui_name="Shaders" value="light_shader" type_name="array_node" />
		<socket name="strength" ui_name="Strength" value="800 800 800 0" type_name="color" />
		<socket name="use_mis" ui_name="Use Mis" value="true" type_name="boolean" />
		<socket name="radius" ui_name="Radius" value="0.2" type_name="float" />
	</pointlight>
	<backgroundlight name="world_light" ui_name="backgroundlight" geometry_type="5">
		<socket name="used_shaders" ui_name="Shaders" value="default_background" type_name="array_node" />
		<socket name="light_type" ui_name="Type" value="background" type_name="enum" />
		<socket name="use_mis" ui_name="Use Mis" value="true" type_name="boolean" />
	</backgroundlight>
	<object name="neural_object" ui_name="Volume">
		<socket name="geometry" ui_name="Geometry" value="neural_volume" type_name="node" />
		<socket name="asset_name" ui_name="Asset Name" value="Volume" type_name="string" />
	</object>
	<object name="key_light_object" ui_name="Light">
		<socket name="geometry" ui_name="Geometry" value="key_light" type_name="node" />
		<socket name="tfm" ui_name="Transform" value="1 0 0 3.0 0 1 0 2.5 0 0 1 5.0" type_name="transform" />
	</object>
	<object name="world_object" ui_name="World">
		<socket name="geometry" ui_name="Geometry" value="world_light" type_name="node" />
	</object>
</cycles>
'''
    with open(args.output, "w") as f:
        f.write(xml)
    open(args.output + ".bin", "wb").close()
    print("wrote", args.output)


if __name__ == "__main__":
    main()
