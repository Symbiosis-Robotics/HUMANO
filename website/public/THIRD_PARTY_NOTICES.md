# External Assets, Attribution & Open-Source Icons

The site includes HUMANO task videos and scene stills rendered with third-party scene and object assets.

## UI icons

The inline line-art UI icons are authored from or adapted to the **Lucide** icon system, including `file-text`, `code-xml`, and `database`. The website intentionally uses `code-xml` rather than the GitHub brand mark for the Code resource.

Project: https://lucide.dev/
Source: https://github.com/lucide-icons/lucide
Licensing: https://github.com/lucide-icons/lucide/blob/main/LICENSE

### ISC License — Lucide

Copyright (c) 2026 Lucide Icons and Contributors

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

Some Lucide icons derive from Feather icons and also carry Feather's MIT license. See the canonical Lucide LICENSE file linked above for the icon-level mapping and complete terms.

## Upstream scene and object libraries

Section 3.1 of the revised HUMANO manuscript credits iTHOR scenes distributed through MolmoSpaces and names Synthesis, MolmoSpaces, and Objaverse as sources for the benchmark’s object pool. See the [iTHOR scene documentation](https://ai2thor.allenai.org/ithor/documentation/scenes/), [MolmoSpaces repository](https://github.com/allenai/molmospaces), and [Objaverse project](https://objaverse.allenai.org/).

### Homepage render records

| Site media | Asset identifiers in the render records |
| --- | --- |
| Hero videos | B-8 episode 5 (FloorPlan307), L-9 episode 7 (FloorPlan4), O-2 episode 49 (FloorPlan308), and C-2 episodes 41 and 17 (FloorPlan17 and FloorPlan8). The B-8 episode uses the Extwin-marked box model `box/box_007`. |
| Overview room stills in `resources/images/scenes/` | Kitchen · FloorPlan10, living room · FloorPlan218, bedroom · FloorPlan325. The overview renderer loads each room from the MolmoSpaces iTHOR background collection. |
| Data Pipeline operator photo (`resources/images/pipeline/pico-operator.png`) | Operator pose reference from Figure 3(a) of the revised HUMANO manuscript. |
| Data Pipeline simulation stills (`mujoco_third.png`, `isaac_third.png`) | HUMANO task L-2, episode 24, frame 1654 in iTHOR FloorPlan8. Both show the same replay from the same third-person camera. |
| HUMANO-Plus A-1 Background in `resources/ood/A-1/` | FloorPlan19 (reference), FloorPlan13, FloorPlan21 and FloorPlan28 (held-out backgrounds). |
| HUMANO-Plus A-1 Distractors in `resources/ood/A-1/distractors-thor.mp4` | `Soap_Bar_1`, `Bowl_1`, `Potato_1`, `Mug_1`, and `Pepper_Shaker_1` from Ai2’s MolmoSpaces THOR object collection. |

**Distractor asset credit.** Allen Institute for AI (Ai2), AI2-THOR and MolmoSpaces. The five THOR meshes were oriented and positioned on the tabletop and rendered into the A-1 replay by HUMANO. The THOR subset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See the [MolmoSpaces dataset and license](https://huggingface.co/datasets/allenai/molmospaces#license).

**Task objects in the A-1 replay.** The five added distractors are credited above. The episode 34 configuration also specifies the task apple (`apple/apple_001`) and basket (`picnic_basket/picnic_basket_009`). The installed source USD layers for both models and the B-8 box carry Extwin copyright and confidentiality notices. Public-display terms for those specific models remain unverified.

The ModelScope and Hugging Face icons are from [LobeHub Icons](https://github.com/lobehub/lobe-icons). The GitHub mark is from [GitHub Octicons](https://github.com/primer/octicons).
