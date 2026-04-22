# 与官方实现的差异 / Bug 记录

## 1. `build_scaling_rotation` 未定义变量

**原代码：**
```python
def build_scaling_rotation(s, R):
    S = torch.diag_embed(s)
    L = R @ L   # L 未定义，应该是 R @ S
    return L
```
**修复：** 用 `R @ S` 构建 transform，再计算 `transform @ transform.T` 得到协方差。

---

## 2. `inverse_sigmoid` 没有数值稳定性保护

**原代码：**
```python
def inverse_sigmoid(x):
    return torch.log(x / (1 - x))  # x=0 或 x=1 时 nan/inf
```
**修复：**
```python
def inverse_sigmoid(x):
    x = x.clamp(min=1e-6, max=1 - 1e-6)
    return torch.log(x / (1 - x))
```

---

## 3. `features` shape 排列错误

**原代码：**
```python
features = torch.zeros((N, 3, sh_dim))   # [N, 3, sh_dim]
features[:, :3, 0] = fused_color
```
**修复：** shape 应为 `[N, sh_dim, 3]`，DC 分量在第 0 个 SH 位置：
```python
features = torch.zeros((N, sh_dim, 3))
features[:, 0, :] = RGB2SH(colors)
```

---

## 4. rotation 初始化用了 `pypose.identity_SO3`

**原代码：**
```python
rots = pypose.identity_SO3(N)  # 依赖 pypose，且输出格式不确定
```
**修复：** 直接用标准四元数 `[w=1, x=0, y=0, z=0]`：
```python
rotation = torch.zeros((N, 4))
rotation[:, 0] = 1.0
```

---

## 5. scaling 初始化公式不对

**原代码：**
```python
scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
# 等价于 0.5 * log(dist2)，但 dist2 是平均平方距离，sqrt 后再 log 不对
```
**修复：** 官方用的是 `log(sqrt(mean_dist2)) = 0.5 * log(mean_dist2)`，直接写清楚：
```python
scaling = 0.5 * torch.log(dist2).unsqueeze(-1).repeat(1, 3)
```

---

## 6. opacity 初始化值不同

**原代码：**
```python
opacities = inverse_sigmoid(0.1 * torch.ones(...))  # 初始 opacity=0.1
```
**修复：** 改为 `0.25`，收敛更快：
```python
opacity = inverse_sigmoid(torch.full((N, 1), 0.25))
```

---

## 7. `scaling` / `opacity` / `features` 属性缺少 `@property` 装饰器

**原代码：**
```python
def scaling(self):   # 普通方法，调用时需要 model.scaling()
    return torch.exp(self.scaling_logit)  # 还有 typo: scaling_logit vs scaling_logits
```
**修复：** 加 `@property`，统一用 `model.scaling` 访问。

---

## 8. `f_rest` 学习率设置错误

**原代码：**
```python
{'params': [self.features_rest], 'lr': training_args.feature_lr / 20.0}
```
**修复：** 官方是除以 4（即 `* 0.25`），不是除以 20：
```python
{"params": [self.features_rest], "lr": cfg.feature_lr * 0.25}
```

---

## 9. rasterizer 返回值 unpack 错误

**原代码：**
```python
color, radii = rasterizer(...)   # rasterizer 实际返回 3 个值
```
**修复：**
```python
color, radii, depth_alpha = rasterizer(...)
```

---

## 10. xyz LR 调度在 optimizer 重建后未恢复

densify/prune 或 opacity reset 后会重建 optimizer，但没有把 xyz LR 恢复到当前 step 对应的值，导致 LR 跳回初始值。

**修复：** 重建 optimizer 后立即调用一次 `update_xyz_lr(optimizer, xyz_scheduler, iteration)`。

---

## 11. `dist_kdtree` 点数不足时 crash

**原代码：**
```python
dists, _ = KDTree(points).query(points, k=4)
dists[:, 1:]  # 如果点数 < 4，index 越界
```
**修复：**
```python
k = min(4, points.shape[0])
dists, _ = KDTree(points).query(points, k=k)
```
