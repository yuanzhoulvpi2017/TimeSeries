# 介绍

这个文件夹 提供了下载个股的数据，代码全部在`downloaddata.py`，使用方法可以查看`股票数据下载demo.ipynb`文件。

# 使用：

## 1. 简单版本

1. stock_id: 是传递的股票的代码，这个股票包含沪深所有**个股**，但是不包括**指数**。在传递参数的时候，需要传递参数类型为字符串，也就是前后加上双引号或者单引号。比如：`'000001'`
2. start_date: 想要的开始时间，需要注意这个时间传递的样式是`yyyy-mm-dd`,格式是文本形式,比如`'2021-01-02'`
3. end_date: 要求和`start_date`一样，在这个基础上，还要求 `start_date` < `end_date`。

返回结果：一个pandas的数据框，这个数据框的列为：'日期', '开盘价', '最高价', '最低价', '收盘价', '涨跌额','涨跌幅(%)', '成交量(手)', '成交金额(万元)', '振幅(%)', '换手率(%)'

```python
# 导入这个文件里面的DlStock函数
from downloaddata import DlStock

# 创建一个新对象
dlstock_object = DlStock(stock_id='600000', start_date='1999-11-10', end_date='2021-10-25')

# 运行这个对象的run函数
data1 = dlstock_object.run()

# 就可以查看数据的前几行了
data1.head()
```

## 2.加入睡眠模式

如果你下载的数据比较多，或者担心频繁的访问服务器，对服务器造成压力，可以设置**休眠参数**。

1. need_sleep: 是开启休眠参数，默认不开启休眠，代表每一轮运行，中间不间断的对服务器访问。
2. sleep_time_range: 当不开启休眠后（也就相当于:`need_sleep=True`,这个时候默认每一轮的休眠时间是在[0,10]
   内整数任意选择一个值，然后程序休眠对应的秒数。当然你也可以自己设置休眠的时间范围，比如`[a,b]`,但是要求a < b.

```python
# 导入这个文件里面的DlStock函数
from downloaddata import DlStock

# 创建一个新对象
dlstock_object = DlStock(stock_id='600000', start_date='1999-11-10', end_date='2021-10-25',
                         need_sleep=True, sleep_time_range=[0, 10])
data1 = dlstock_object.run()
data1.head()
```

## 3. 并行下载

如果你觉得下载数据太慢了，希望提高并行的数量，那么你可以设置`njobs`参数。

1. njobs: 这个参数默认是10，代表最高同时有10个下载器在分批下载你的数据。如果你觉得10个太小了，可以设置更多，比如12，16，20等，只要是正整数即可。

```python
# 导入这个文件里面的DlStock函数
from downloaddata import DlStock

# 创建一个新对象
dlstock_object = DlStock(stock_id='600000', start_date='1999-11-10', end_date='2021-10-25',
                         njobs=10)
data1 = dlstock_object.run()
data1.head()
```

# 联系我

如果有更多使用问题，可以联系我：<img height="512" src="https://gitee.com/yuanzhoulvpi/time_series/raw/master/%E4%B8%8B%E8%BD%BD%E4%B8%AA%E8%82%A1%E8%82%A1%E7%A5%A8%E6%95%B0%E6%8D%AE/images/pypi_timeseries_wechat_group.JPG" title="python与时间序列微信群" width="400"/>