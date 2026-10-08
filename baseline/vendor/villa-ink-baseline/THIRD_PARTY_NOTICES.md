# Third-party source notices

The unmodified source files in this directory were taken from [ScrollPrize/villa](https://github.com/ScrollPrize/villa/tree/e0bbb8b40a2db58b1d71864f286eb85717e59e64) at commit `e0bbb8b40a2db58b1d71864f286eb85717e59e64`. The retained [MIT license](UPSTREAM-LICENSE.txt) states **Copyright (c) 2024 Vesuvius Challenge**. [source-manifest.json](source-manifest.json) preserves each upstream path, Git blob identity and SHA-256. Neither that manifest nor the source files is changed by these additional notices.

Several modules explicitly identify other projects as copied, adapted or cited sources. Those projects' Apache 2.0 license texts and copyright notices are retained below in addition to villa's MIT notice. These notices do not replace the underlying licenses or declare every third-party component to be MIT.

| Project | Source attribution in this snapshot | Retained license |
| --- | --- | --- |
| [MIC-DKFZ/dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures) | `vesuvius/models/build/encoder.py` and `decoder.py` say they are copied from this project; `resblocks.py`, `models/utilities/utils.py` and `build/transformers/patch_encode_decode.py` cite related source paths. | [Apache-2.0, full text](ADDITIONAL-LICENSES/DYNAMIC-NETWORK-ARCHITECTURES-LICENSE.txt) |
| [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) | `vesuvius/models/utils.py` and `models/utilities/utils.py` cite this project; `build/build_network_from_config.py` identifies the nnU-Net architectural inspiration. | [Apache-2.0, full text](ADDITIONAL-LICENSES/NNUNET-LICENSE.txt) |
| [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) | `vesuvius/models/build/mednext/blocks.py` and `v1.py` identify adaptation from the official MedNeXt repository. | [Apache-2.0, full text](ADDITIONAL-LICENSES/MEDNEXT-LICENSE.txt) |

The retained license appendices contain these copyright notices, reproduced with their original brackets:

- Dynamic Network Architectures: `Copyright [2022] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]`.
- nnU-Net: `Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]`.
- MedNeXt: `Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]`.

The upstream LICENSE files were downloaded over verified HTTPS on 8 October 2026 and preserved byte for byte, including the license terms and their appendices. Each URL is pinned to the most recent upstream commit that changed that LICENSE file. The source comments identify ancestry but do not establish the exact original revision from which each derivative was made.

| License source URL | Retrieved UTC | SHA-256 |
| --- | --- | --- |
| [dynamic-network-architectures / LICENSE](https://raw.githubusercontent.com/MIC-DKFZ/dynamic-network-architectures/97b027c8d914818bce829c2d9a3f9547c1343d49/LICENSE) | `2026-10-08T08:11:02.460461+00:00` | `7b26c3dffb131e8206336ecc7680a0d0080007cf49b7de7ad64ae936e78a1ce4` |
| [nnUNet / LICENSE](https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/ad3b4f9295c278b3467a2adc5f4ee93effb128f2/LICENSE) | `2026-10-08T08:11:02.264784+00:00` | `ff93b6fdf1d59f7c48ca7f44f2d0db47bb7743a1c6fa0f4c5f4d7809f78ce70e` |
| [MedNeXt / LICENSE](https://raw.githubusercontent.com/MIC-DKFZ/MedNeXt/7fdbd4ab669400c721a4fef8df547f99bdb34e39/LICENSE) | `2026-10-08T08:11:02.314702+00:00` | `7da9f2ddeda836a891f537104530230a7b2318f61ea4c57b05e3dd46a9ae27fd` |

The [machine-readable acquisition record](ADDITIONAL-LICENSES/sources.json) also records byte lengths and commit identifiers. Preserve this file and the complete license texts when redistributing this source snapshot. Installed dependencies such as PyTorch, timm and NumPy retain their own licenses; their package source is not vendored here. Model weights, scan data, annotations and published illustrations have separate terms and are not licensed by these code notices.
