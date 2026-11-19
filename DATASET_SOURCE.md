# Dataset source and license

This project uses the public **iBean / Beans** image dataset created by the Makerere AI Lab in collaboration with the National Crops Resources Research Institute (NaCRRI), Uganda.

- Original project: https://github.com/AI-Lab-Makerere/ibean
- Primary course archive source: https://huggingface.co/datasets/AI-Lab-Makerere/beans
- Historical public archive fallback: https://storage.googleapis.com/ibeans/
- Original release: 20 January 2020
- License: MIT
- Original project reports 1,296 images. The exact split archives used in this lab contain 1,295 images: 1,034 train, 133 validation, 128 test.
- Classes: `angular_leaf_spot`, `bean_rust`, `healthy`

The downloader tries the Hugging Face dataset mirror first, can fall back to the historical Google Storage archive, and always verifies the published SHA-256 checksum before extraction. Therefore, the dataset identity does not depend on which mirror serves the bytes.

## Attribution

Makerere AI Lab, *Bean Disease Dataset*, 2020.

## License notice

MIT License

Copyright (c) 2020 AIR Lab Makerere University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
