## 程序目的

将一个AI生成CAD中面几何提取的项目结果可视化。

## 输入

### 模型和目录

工作目录默认是 F:\newdata
（可以在代码中修改）

工作目录结构：包含一系列子文件夹，每个子文件夹包含一系列格式为.step 的 cad模型文件。

使用OCC库来和*.step模型文件交互。

### NPZ 文件 ( `{file}_{left_face_tag}_{right_face_tag}_result.npz` )

左面是tag=193这个face；
右面是tag=168这个face；
因此可以读入这两个face的数据；


| 字段              | Shape      | dtype   | 值域     | 说明                                |
| --------------- | ---------- | ------- | ------ | --------------------------------- |
| `query_points`  | (20480, 3) | float32 | [0, 1] | 查询点 xyz 坐标                        |
| `offset_pred`   | (20480,)   | float32 | [0, 1] | Stage2 预测 offset (0=靠近S1, 1=靠近S2) |
| `offset_gt`     | (20480,)   | float32 | [0, 1] | 真值 offset                         |
| `validity_pred` | (20480,)   | float32 | {0, 1} | Stage1 预测有效性 (1=有效/在S1S2之间)       |
| `validity_gt`   | (20480,)   | float32 | {0, 1} | 真值有效性 (1=有效)                      |
| `points_S1`     | (1024, 3)  | float32 | [0, 1] | S1 曲面点云                           |
| `points_S2`     | (1024, 3)  | float32 | [0, 1] | S2 曲面点云                           |


### JSON 文件 ( `{folder}_{file}_result.json` ) — 元数据

json文件现在还用不到，请忽略

```json
{
  "sample_idx": 0,
  "num_points": 20480
}
```

### 加载方式

```python
import numpy as np

data = np.load("26_26_193_168_result.npz")
query_points = data["query_points"]  # (N, 3)
offset_pred = data["offset_pred"]    # (N,)
validity_pred = data["validity_pred"]# (N,) binary
```

## 输出/功能

### 功能1：显示模型

在窗口中显示当前的.step模型。

需要能在窗口中调整摄像机的位姿，也需要将摄像机的所有可用参数，比如fov等开放到代码中。

需要将显示模型的材质参数开放到代码中，方便我修改。

初始需要的材质：半透明。

不需要实时渲染，可以是可交互渲染（比如，在摄像机不移动时的几秒钟内逐渐收敛，在移动摄像机的时候重来）。