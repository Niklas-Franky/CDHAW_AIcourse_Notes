```
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4)):设置画布大小
plt.plot(x, y)：画折线
plt.scatter(x, y)：画散点
plt.bar(x, y)：画柱状图
plt.hist(x)：画直方图
plt.title('标题')：加标题
plt.xlabel('横坐标名')：加横轴标签
plt.ylabel('纵坐标名')：加纵轴标签
plt.xlim([a, b])：限制x轴范围
plt.ylim([a, b])：限制y轴范围
plt.grid(True)：加网格
plt.savefig('图片名.png')：保存图片
plt.show()：显示图片窗口（jupyter里不加也能自动显示）
```
# plt.scatter 相关参数
| 参数         | 作用     | 举例说明                                    |
| ---------- | ------ | --------------------------------------- |
| x          | 横坐标数据  | 必填，比如`x=[1,2,3]`                        |
| y          | 纵坐标数据  | 必填，比如`y=[3,4,5]`                        |
| c          | 点的颜色   | `c='r'`（红色），`c='b'`（蓝色），也可以传一组颜色（分组染色）  |
| marker     | 点的形状   | `'o'`圆点，`'s'`方块，`'^'`三角，`'x'`叉号，`'+'`加号 |
| s          | 点的大小   | `s=100`，值越大点越大                          |
| alpha      | 透明度    | `alpha=0.5`，0完全透明，1不透明                  |
| label      | 图例标签   | `label='类别1'`，配合`plt.legend()`显示        |
| edgecolors | 点的边框颜色 | `edgecolors='k'`（黑色边）                   |
| linewidths | 点的边框粗细 | `linewidths=2`                          |
| cmap       | 渐变色带   | `cmap='viridis'`，适合c是一组数字时              |


# plt.plot怎么写
plt.plot(X , y , 'm-' , linewidth = , label = '')
plt.legend( loc = 'lower left'  ) 有label就要设置一下legend图例的位置
'm-'是style样式字符串 还有很多，见下面
### 颜色
| 代码    | 颜色           |
| ----- | ------------ |
| `'b'` | 蓝色 (blue)    |
| `'g'` | 绿色 (green)   |
| `'r'` | 红色 (red)     |   
| `'c'` | 青色 (cyan)    |
| `'m'` | 品红 (magenta) |
| `'y'` | 黄色 (yellow)  |
| `'k'` | 黑色 (black)   |
| `'w'` | 白色 (white)   |
### 点
| 符号    | 形状   |
| ----- | ---- |
| `'.'` | 像素点  |
| `','` | 更小的点 |
| `'o'` | 圆圈   |
| `'v'` | 倒三角  |
| `'^'` | 上三角  |
| `'s'` | 正方形  |
| `'D'` | 菱形   |
| `'*'` | 星号   |
| `'x'` | 叉号   |
| `'+'` | 加号   |
### 线
| 符号     | 样式  |
| ------ | --- |
| `'-'`  | 实线  |
| `'--'` | 虚线  |
| `'-.'` | 点划线 |
| `':'`  | 点线  |
| `''`   | 不连线 |
