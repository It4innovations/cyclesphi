/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "app/cycles_xml_nodes.h"

#include "device/denoise.h"

#include "scene/background.h"
#include "scene/camera.h"
#include "scene/film.h"
#include "scene/hair.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/particles.h"
#include "scene/pass.h"
#include "scene/pointcloud.h"
#include "scene/shader.h"
#include "scene/shader_nodes.h"
#include "scene/volume.h"

#include "session/buffers.h"

#include "util/thread.h"

CCL_NAMESPACE_BEGIN

void register_all_nodes() 
{
  static thread_mutex mutex;
  static bool initialized = false;
  thread_scoped_lock lock(mutex);
  if (initialized) {
    return;
  }

  // Call all node type getters (each triggers NodeType::add(...) once).
  
  // Shader nodes
  (void)ImageTextureNode::get_node_type();
  (void)EnvironmentTextureNode::get_node_type();
  (void)SkyTextureNode::get_node_type();
  (void)GradientTextureNode::get_node_type();
  (void)NoiseTextureNode::get_node_type();
  (void)GaborTextureNode::get_node_type();
  (void)VoronoiTextureNode::get_node_type();
  (void)IESLightNode::get_node_type();
  (void)WhiteNoiseTextureNode::get_node_type();
  (void)WaveTextureNode::get_node_type();
  (void)MagicTextureNode::get_node_type();
  (void)CheckerTextureNode::get_node_type();
  (void)BrickTextureNode::get_node_type();
  (void)NormalNode::get_node_type();
  (void)MappingNode::get_node_type();
  (void)RGBToBWNode::get_node_type();
  (void)MetallicBsdfNode::get_node_type();
  (void)GlossyBsdfNode::get_node_type();
  (void)GlassBsdfNode::get_node_type();
  (void)RefractionBsdfNode::get_node_type();
  (void)ToonBsdfNode::get_node_type();
  (void)SheenBsdfNode::get_node_type();
  (void)DiffuseBsdfNode::get_node_type();
  (void)PrincipledBsdfNode::get_node_type();
  (void)TranslucentBsdfNode::get_node_type();
  (void)TransparentBsdfNode::get_node_type();
  (void)RayPortalBsdfNode::get_node_type();
  (void)SubsurfaceScatteringNode::get_node_type();
  (void)EmissionNode::get_node_type();
  (void)BackgroundNode::get_node_type();
  (void)HoldoutNode::get_node_type();
  (void)AmbientOcclusionNode::get_node_type();
  (void)AbsorptionVolumeNode::get_node_type();
  (void)ScatterVolumeNode::get_node_type();
  (void)VolumeCoefficientsNode::get_node_type();
  (void)PrincipledVolumeNode::get_node_type();
  (void)PrincipledHairBsdfNode::get_node_type();
  (void)HairBsdfNode::get_node_type();
  (void)GeometryNode::get_node_type();
  (void)TextureCoordinateNode::get_node_type();
  (void)UVMapNode::get_node_type();
  (void)LightPathNode::get_node_type();
  (void)LightFalloffNode::get_node_type();
  (void)ObjectInfoNode::get_node_type();
  (void)ParticleInfoNode::get_node_type();
  (void)HairInfoNode::get_node_type();
  (void)PointInfoNode::get_node_type();
  (void)VolumeInfoNode::get_node_type();
  (void)VertexColorNode::get_node_type();
  (void)ValueNode::get_node_type();
  (void)ColorNode::get_node_type();
  (void)AddClosureNode::get_node_type();
  (void)MixClosureNode::get_node_type();
  (void)MixClosureWeightNode::get_node_type();
  (void)InvertNode::get_node_type();
  (void)MixNode::get_node_type();
  (void)MixColorNode::get_node_type();
  (void)MixFloatNode::get_node_type();
  (void)MixVectorNode::get_node_type();
  (void)MixVectorNonUniformNode::get_node_type();
  (void)CombineColorNode::get_node_type();
  (void)CombineXYZNode::get_node_type();
  (void)GammaNode::get_node_type();
  (void)BrightContrastNode::get_node_type();
  (void)SeparateColorNode::get_node_type();
  (void)SeparateXYZNode::get_node_type();
  (void)HSVNode::get_node_type();
  (void)AttributeNode::get_node_type();
  (void)CameraNode::get_node_type();
  (void)FresnelNode::get_node_type();
  (void)LayerWeightNode::get_node_type();
  (void)WireframeNode::get_node_type();
  (void)WavelengthNode::get_node_type();
  (void)BlackbodyNode::get_node_type();
  (void)OutputNode::get_node_type();
  (void)MapRangeNode::get_node_type();
  (void)VectorMapRangeNode::get_node_type();
  (void)ClampNode::get_node_type();
  (void)OutputAOVNode::get_node_type();
  (void)MathNode::get_node_type();
  (void)VectorMathNode::get_node_type();
  (void)VectorRotateNode::get_node_type();
  (void)VectorTransformNode::get_node_type();
  (void)BumpNode::get_node_type();
  (void)RGBCurvesNode::get_node_type();
  (void)VectorCurvesNode::get_node_type();
  (void)FloatCurveNode::get_node_type();
  (void)RGBRampNode::get_node_type();
  (void)SetNormalNode::get_node_type();
  (void)NormalMapNode::get_node_type();
  (void)RadialTilingNode::get_node_type();
  (void)TangentNode::get_node_type();
  (void)BevelNode::get_node_type();
  (void)DisplacementNode::get_node_type();
  (void)VectorDisplacementNode::get_node_type();
  (void)RaycastNode::get_node_type();

  // Scene nodes
  (void)DenoiseParams::get_node_type();
  (void)BufferPass::get_node_type();
  (void)BufferParams::get_node_type();
  (void)Volume::get_node_type();
  (void)Shader::get_node_type();
  (void)PointCloud::get_node_type();
  (void)Pass::get_node_type();
  (void)ParticleSystem::get_node_type();
  (void)Object::get_node_type();
  (void)Mesh::get_node_type();
  (void)PointLight::get_node_type();
  (void)SpotLight::get_node_type();
  (void)AreaLight::get_node_type();
  (void)SunLight::get_node_type();
  (void)BackgroundLight::get_node_type();
  (void)Integrator::get_node_type();
  (void)Hair::get_node_type();
  (void)Film::get_node_type();
  (void)Camera::get_node_type();
  (void)Background::get_node_type();

  initialized = true;
}

CCL_NAMESPACE_END
