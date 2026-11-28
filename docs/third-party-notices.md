# Third-party notices

## SIMPLE

Parts of HUMANO derive from SIMPLE. The retained source headers credit the following contributors. The exact upstream import revision has not been established. See the [SIMPLE project](https://github.com/physical-superintelligence-lab/SIMPLE).

```text
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Included components

| Component | License information |
| --- | --- |
| Unitree SDK2 Python | [BSD 3-Clause text and attribution](../third_party/unitree_sdk2_python/LICENSE). |
| cuRobo | This copy carries the [NVIDIA research and evaluation license](../third_party/curobo/LICENSE), including its noncommercial restriction. See the separate [asset terms](../third_party/curobo/LICENSE_ASSETS). |
| GR00T N1.7, LeRobot runtime and Psi0 baselines | See the Apache licenses in [GR00T](../baselines/gr00t_n17/LICENSE), [LeRobot](../baselines/lerobot_runtime/LICENSE) and [Psi0](../baselines/psi0/LICENSE). Check individual components and weights separately. |
| decoupled_wbc, gear_sonic, gear_sonic_deploy source code | See the [NVIDIA WholeBodyControl license guide](https://nvlabs.github.io/GR00T-WholeBodyControl/resources/license.html). The exact source import revisions remain to be verified. |
| NVIDIA model weights | The [model notice](../NOTICE.txt) lists all three bundled weights. The [WBC model agreement](../third_party/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/NVIDIA%20Open%20Model%20License) covers Balance and Walk. The [SONIC model agreement](../third_party/gear_sonic_deploy/policy/release/LICENSE-model.txt) covers the [official decoder](https://huggingface.co/nvidia/GEAR-SONIC/blob/0af0cb7568257b04a8569b9c76f326e707b122fc/model_decoder.onnx). |
| Unitree C++ SDK deployment binaries (aarch64 and x86_64) | `libunitree_sdk2.a` ships under the [Unitree BSD 3-Clause license](../third_party/gear_sonic_deploy/thirdparty/unitree_sdk2/LICENSE). HUMANO distributes `libddsc.so` and `libddscxx.so` under the EDL-1.0 option in the [Cyclone DDS](../third_party/gear_sonic_deploy/thirdparty/unitree_sdk2/licenses/eclipse-cyclonedds/cyclonedds/LICENSE) and [Cyclone DDS C++](../third_party/gear_sonic_deploy/thirdparty/unitree_sdk2/licenses/eclipse-cyclonedds/cyclonedds-cxx/LICENSE) licenses, respectively. See the [bundled dependency licenses](../third_party/gear_sonic_deploy/thirdparty/unitree_sdk2/licenses/). |

## Downloaded resources

Robot meshes, model weights, materials, scenes and object models carry their own terms. The project's license grants no additional rights over those resources. Revo2 model redistribution remains unconfirmed for the pinned BrainCo revision. HSSD and GraspNet have noncommercial restrictions. Mixed object collections require per-asset attribution and license checks. Consult the original asset packages for BlenderKit and NVIDIA vMaterials distribution terms. This local preparation does not establish clearance to redistribute every referenced resource.
