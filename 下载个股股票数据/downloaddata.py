import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import random


class DlStock(object):
    """
    ==================================================================================================================
    Params：
    下载个股股票数据：沪深的个股数据，但是不包含指数数据。
    stock_id:股票id，必须设置为字符串。
    start_date:下载股票的时间段的开始时间，必须为字符串，样式为yyyy-mm-dd。
    end_date:下载股票的时间段的结束时间，必须为字符串，样式为yyyy-mm-dd,且要大于start_date。
    njobs:设置并行参数，可以提高下载数据的速度，继续为整数，默认数值为10。
    need_sleep：为了防止频繁的下载数据，造成对服务器有害影响，防止ip被服务器封锁，
                这里设置每一个线程可以休息几秒，时间范围为sleep_time_range,
                默认每个线程不需要休息几秒。
    sleep_time_range:配合need_sleep参数一起使用，默认休息时间是0到10秒内容，必须为一个长度为2的列表。
                代表每一轮的休眠时间在[a,b]范围内。
    ==================================================================================================================
    结果：
        最终返回一个下载结果：这个股票在改时间范围内的几个指标：
        '日期', '开盘价', '最高价', '最低价', '收盘价', '涨跌额','涨跌幅(%)', '成交量(手)', '成交金额(万元)', '振幅(%)', '换手率(%)'

    """

    def __init__(self, stock_id: str = '000001',
                 start_date: str = '2021-01-01',
                 end_date: str = '2021-09-01',
                 njobs: int = 10,
                 need_sleep: bool = False,
                 sleep_time_range: list = [0, 10]):
        self.njobs = njobs
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self.empty_data = pd.DataFrame(columns=['日期', '开盘价', '最高价', '最低价', '收盘价', '涨跌额',
                                                '涨跌幅(%)', '成交量(手)', '成交金额(万元)', '振幅(%)', '换手率(%)'])
        self.day_index = None
        self.need_sleep = need_sleep
        self.sleep_time_range = sleep_time_range
        self.finaldata = None

    def generat_index_data(self):
        day_index = pd.DataFrame({'day': pd.date_range(start=self.start_date, end=self.end_date)})
        day_index['year'] = day_index['day'].apply(lambda x: x.year)
        day_index['season'] = day_index['day'].apply(lambda x: ((x.month - 1) // 3) + 1)
        day_index = day_index.drop_duplicates(subset=['year', 'season']).drop(columns=['day']).reset_index(drop=True)
        self.day_index = day_index
        self.index_range_max = self.day_index.shape[0]

    def dl_stock_epoch_data(self, stock_id: str, year: int, season: int) -> pd.DataFrame:
        temp_url = f"https://quotes.money.163.com/trade/lsjysj_{stock_id}.html?year={year}&season={season}"
        try:
            data = pd.read_html(temp_url)[3]
        except Exception as e:
            data = self.empty_data
        finally:
            return data

    def dl_data_by_index(self, index: int) -> pd.DataFrame:
        if self.need_sleep:
            time.sleep(random.randint(*self.sleep_time_range))

        index_temp_data = self.day_index.iloc[index]
        data = self.dl_stock_epoch_data(stock_id=self.stock_id, year=index_temp_data.year,
                                        season=index_temp_data.season)

        return data

    def run(self) -> pd.DataFrame:
        self.generat_index_data()
        result_list = Parallel(n_jobs=self.njobs)(
            delayed(self.dl_data_by_index)(index) for index in tqdm(range(self.index_range_max), desc="downloading..."))
        self.finaldata = pd.concat(result_list).sort_values(by=['日期']).reset_index(drop=True)

        return self.finaldata


if __name__ == '__main__':
    """
    检验这个函数是否有错误
    """
    dlstock1 = DlStock(
        stock_id='600612',
        start_date='1990-08-01',
        end_date='2021-10-25')
    data = dlstock1.run()
    print(data.head())
