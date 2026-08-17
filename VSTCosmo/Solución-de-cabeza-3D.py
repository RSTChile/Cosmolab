import bpy

# Limpiar escena
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 1. ESFERA BASE
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=64, ring_count=32)
esfera = bpy.context.active_object
esfera.name = "Cabeza"
mat_gris = bpy.data.materials.new(name="GrisMate")
mat_gris.use_nodes = True
mat_gris.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.72, 0.75, 0.78, 1) # #B8C0C8
mat_gris.node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0.8 # Roughness
esfera.data.materials.append(mat_gris)

# 2. CUENCAS - Boolean
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.28, location=(-0.32, 0.18, 0.85))
cuenca_l = bpy.context.active_object
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.28, location=(0.32, 0.18, 0.85))
cuenca_r = bpy.context.active_object

bool_l = esfera.modifiers.new(name="Cuenca_L", type='BOOLEAN')
bool_l.operation = 'DIFFERENCE'
bool_l.object = cuenca_l
bool_r = esfera.modifiers.new(name="Cuenca_R", type='BOOLEAN')
bool_r.operation = 'DIFFERENCE'
bool_r.object = cuenca_r

bpy.context.view_layer.objects.active = esfera
bpy.ops.object.modifier_apply(modifier="Cuenca_L")
bpy.ops.object.modifier_apply(modifier="Cuenca_R")
bpy.data.objects.remove(cuenca_l)
bpy.data.objects.remove(cuenca_r)

# 3. OJOS
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.22, location=(-0.32, 0.18, 0.90))
ojo_l = bpy.context.active_object
ojo_l.name = "Ojo_L"
mat_blanco = bpy.data.materials.new(name="Blanco")
mat_blanco.use_nodes = True
mat_blanco.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.94, 0.94, 0.94, 1)
ojo_l.data.materials.append(mat_blanco)

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.22, location=(0.32, 0.18, 0.90))
ojo_r = bpy.context.active_object
ojo_r.name = "Ojo_R"
ojo_r.data.materials.append(mat_blanco)

# 4. IRIS - Shrinkwrap sobre el ojo
bpy.ops.mesh.primitive_circle_add(radius=0.13, vertices=32, location=(-0.32, 0.18, 1.11))
iris_l = bpy.context.active_object
iris_l.name = "Iris_L"
mat_azul = bpy.data.materials.new(name="Azul")
mat_azul.use_nodes = True
mat_azul.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0, 0.4, 1, 1)
mat_azul.node_tree.nodes["Principled BSDF"].inputs[19].default_value = (0, 0.4, 1, 1) # Emission
mat_azul.node_tree.nodes["Principled BSDF"].inputs[20].default_value = 0.3 # Emission strength
iris_l.data.materials.append(mat_azul)
shrink_l = iris_l.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
shrink_l.target = ojo_l
shrink_l.wrap_method = 'PROJECT'
shrink_l.use_negative_direction = True

bpy.ops.mesh.primitive_circle_add(radius=0.13, vertices=32, location=(0.32, 0.18, 1.11))
iris_r = bpy.context.active_object
iris_r.name = "Iris_R"
iris_r.data.materials.append(mat_azul)
shrink_r = iris_r.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
shrink_r.target = ojo_r
shrink_r.wrap_method = 'PROJECT'
shrink_r.use_negative_direction = True

# 5. OREJAS
bpy.ops.mesh.primitive_torus_add(major_radius=0.18, minor_radius=0.06, location=(-1.02, 0, 0))
oreja_l = bpy.context.active_object
oreja_l.name = "Oreja_L"
oreja_l.rotation_euler[1] = 1.5708
oreja_l.data.materials.append(mat_gris)

bpy.ops.mesh.primitive_torus_add(major_radius=0.18, minor_radius=0.02, location=(-1.02, 0, 0))
anillo_l = bpy.context.active_object
anillo_l.name = "Anillo_L"
mat_cyan = bpy.data.materials.new(name="Cyan")
mat_cyan.use_nodes = True
mat_cyan.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0, 0.87, 1, 1)
mat_cyan.node_tree.nodes["Principled BSDF"].inputs[19].default_value = (0, 0.87, 1, 1)
mat_cyan.node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1.0
anillo_l.data.materials.append(mat_cyan)
anillo_l.rotation_euler[1] = 1.5708

bpy.ops.mesh.primitive_torus_add(major_radius=0.18, minor_radius=0.06, location=(1.02, 0, 0))
oreja_r = bpy.context.active_object
oreja_r.name = "Oreja_R"
oreja_r.rotation_euler[1] = 1.5708
oreja_r.data.materials.append(mat_gris)

bpy.ops.mesh.primitive_torus_add(major_radius=0.18, minor_radius=0.02, location=(1.02, 0, 0))
anillo_r = bpy.context.active_object
anillo_r.name = "Anillo_R"
mat_rojo = bpy.data.materials.new(name="Rojo")
mat_rojo.use_nodes = True
mat_rojo.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1, 0.2, 0.33, 1)
mat_rojo.node_tree.nodes["Principled BSDF"].inputs[19].default_value = (1, 0.2, 0.33, 1)
mat_rojo.node_tree.nodes["Principled BSDF"].inputs[20].default_value = 1.0
anillo_r.data.materials.append(mat_rojo)
anillo_r.rotation_euler[1] = 1.5708

# 6. UNIR TODO
bpy.ops.object.select_all(action='DESELECT')
esfera.select_set(True)
ojo_l.select_set(True)
ojo_r.select_set(True)
iris_l.select_set(True)
iris_r.select_set(True)
oreja_l.select_set(True)
anillo_l.select_set(True)
oreja_r.select_set(True)
anillo_r.select_set(True)
bpy.context.view_layer.objects.active = esfera
bpy.ops.object.join()
esfera.name = "VST_Cabeza"

# 7. EXPORTAR
bpy.ops.export_scene.gltf(filepath="/ruta/vst_cabeza_v2.glb", export_format='GLB')