# YOLO26s HTP C++ Demo

该 Demo 使用 `yolo26s_split_quantized.dlc` 的独立输出：

```text
output_boxes  [1,4,21504]
output_scores [1,1,21504]
```

C++ 程序负责：

1. 读取 JPG/PNG；
2. letterbox 到 `1024x1024`，使用数值 114 填充；
3. BGR 转 RGB，生成 `[1,3,1024,1024]` float32 `[0,1]` 输入；
4. 调用 `qnn-net-run + libQnnHtp.so + libQnnModelDlc.so`；
5. 解码 `output_boxes`、`output_scores`；
6. 执行置信度过滤和 NMS；
7. 将框从 letterbox 坐标还原到原图；
8. 绘制检测结果并保存 JPG。

## 构建

如果容器内尚未安装构建依赖：

```bash
apt-get update
apt-get install -y --no-install-recommends \
  cmake \
  libopencv-core-dev \
  libopencv-imgproc-dev \
  libopencv-imgcodecs-dev
```

在 `qairt_run` 容器中执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qairt_310
source /opt/qairt-embedded/bin/envsetup.sh

cd /workspace/yolo26s_cpp_demo
cmake -S . -B build
cmake --build build -j
```

## 运行

```bash
./build/yolo26s_htp_demo \
  --model /workspace/yolo26s_split_quantized.dlc \
  --image "/workspace/DJI_20251029164538_0027_V_   .JPG" \
  --output /workspace/yolo26s_cpp_result.jpg
```

默认参数：

```text
backend          /opt/qairt-embedded/lib/x86_64-linux-clang/libQnnHtp.so
dlc-loader       /opt/qairt-embedded/lib/x86_64-linux-clang/libQnnModelDlc.so
score-threshold  0.60
nms-iou          0.45
```

可用参数：

```text
--model
--image
--output
--backend
--dlc-loader
--qnn-net-run
--score-threshold
--nms-iou
```

`qnn-net-run` 默认从 `PATH` 查找。若未激活 QAIRT 环境，可通过
`--qnn-net-run /opt/qairt-embedded/bin/x86_64-linux-clang/qnn-net-run`
指定绝对路径。
