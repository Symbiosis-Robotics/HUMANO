# Install simulation resources

HUMANO installs simulation resources under `data/resources/`. For offline training on the recorded videos and demonstrations, download the [task archive](../../data/README.md). For simulation, also install the resources used by the task and its robot configuration. The task ZIP supplies the frozen evaluation bindings.

## Resource archives

| Archive | What it installs |
| --- | --- |
| `molmospaces-ithor.zip` | The scene catalog, 90 iTHOR FloorPlans covering 93 room entries, and their referenced THOR models and textures under `scenes/` and `objects/thor/`. |
| `rigid-objects.zip` | 65 rigid object models under `objects/rigid/`: 31 apples, 30 plates and four pens, with the included visual and collision files. |
| `unitree-g1-urdf.zip` | Three G1 URDFs (29-DoF body, Dex3 and whole-body Dex3) and the 52 STL meshes they reference under `robots/g1/`. |
| `humano_material.zip` | The HUMANO matte-brown table material under `materials/vMaterials_2/SolidColor/`. It imports NVIDIA's MDL core definitions. |
| `polyfork-kitchen-bin.zip` | The optional Polyfork Kitchen Bin source GLB and attribution under `objects/source/polyfork/kitchen_bin/`. The GLB has no simulated joint or collision configuration. |

Download the selected ZIPs from the `resources/` folder of the [HUMANO dataset on ModelScope](https://www.modelscope.cn/datasets/SymbiosisRobotics/HUMANO). From the HUMANO checkout, install the iTHOR scenes with:

```bash
archive='resources/molmospaces-ithor.zip'
modelscope download --repo-type dataset SymbiosisRobotics/HUMANO "$archive" --local-dir downloads
mkdir -p data/resources
unzip "downloads/$archive" -d data/resources/
test -f data/resources/scenes/molmo_ithor_ithor120_background/scene_catalog.json
```

For a browser download, use the saved ZIP path in the same extraction command. Extract each selected archive to the same destination. Each ZIP carries its own file list, sources, revisions and attribution under `package_info/`.

## Additional dependencies

The archives cover the selections in the table. Depending on the task, evaluation can also require other task objects, a simulation-ready robot MJCF or USD, the SONIC decoder, or a material library and the matching `material_split.npy`. Use the [installation guide](../install.md) for SONIC and the [data layout guide](../../data/README.md) for expected paths.

- For HSSD scenes and GraspNet objects or grasps, obtain the required files from their upstream distributions under the applicable terms. [HSSD](https://huggingface.co/datasets/hssd/hssd-hab) · [GraspNet](https://graspnet.net/datasets.html)
- For tasks using NVIDIA materials, the reference installation uses vMaterials 2.2.1. Install the library and use the matching HUMANO material assignments. TensorRT is a separate dependency for configurations that select it. [NVIDIA vMaterials](https://developer.nvidia.com/vmaterials) · [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt/download)
