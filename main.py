import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import time
from pytz import timezone

import settings


# 로그 기록
log_dir = os.path.join(settings.BASE_DIR, 'logs')
timestr = settings.get_time_str()
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
file_handler = logging.FileHandler(filename=os.path.join(
    log_dir, "{}.log".format(timestr)), encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(format="%(message)s",
                    handlers=[file_handler, stream_handler], level=logging.DEBUG)


import data_manager
from agent import Agent
from learners import PolicyLearner, A2CLearner, A3CLearner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_codes', nargs='+')
    parser.add_argument('--ver', default=datetime.fromtimestamp(time.time(), timezone('Asia/Seoul')).strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--rl_method', default='pg', help='Should be one of pg, a2c, and a3c')
    parser.add_argument('--net', default='lstm', help='dnn, lstm')
    parser.add_argument('--n_steps', default=5)
    args = parser.parse_args()
    
    # 모델 경로 준비
    model_dir = os.path.join(settings.BASE_DIR, 'models')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    policy_network_path = os.path.join(model_dir, 'p_%s.h5' % args.ver)
    value_network_path = os.path.join(model_dir, 'v_%s.h5' % args.ver)

    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []

    for stock_code in args.stock_codes:
        # 주식 데이터 준비
        chart_data = data_manager.load_chart_data(
            os.path.join(settings.BASE_DIR,
                        'data/chart_data/{}.csv'.format(stock_code)))
        prep_data = data_manager.preprocess(chart_data)
        training_data = data_manager.build_training_data(prep_data)

        # 기간 필터링
        training_data = training_data[(training_data['date'] >= '2017-01-01') &
                                    (training_data['date'] <= '2017-12-31')]
        training_data = training_data.dropna()

        # 차트 데이터 분리
        features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
        chart_data = training_data[features_chart_data]

        # 학습 데이터 분리
        features_training_data = [
            'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
            'close_lastclose_ratio', 'volume_lastvolume_ratio',
            'close_ma5_ratio', 'volume_ma5_ratio',
            'close_ma10_ratio', 'volume_ma10_ratio',
            'close_ma20_ratio', 'volume_ma20_ratio',
            'close_ma60_ratio', 'volume_ma60_ratio',
            'close_ma120_ratio', 'volume_ma120_ratio',
            # 'pbr', 'per', 'roe'
        ]
        training_data = training_data[features_training_data]

        # 최소/최대 투자 단위 설정
        min_trading_unit = max(int(100000 / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = max(int(300000 / chart_data.iloc[-1]['close']), 1)

        # 강화학습 시작
        learner = None
        if args.rl_method == 'pg':
            learner = PolicyLearner(
                stock_code=stock_code, chart_data=chart_data, training_data=training_data,
                min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit, delayed_reward_threshold=.05,
                net='lstm', n_steps=5, lr=.01)
        elif args.rl_method == 'a2c':
            learner = A2CLearner(
                stock_code=stock_code, chart_data=chart_data, training_data=training_data,
                min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit, delayed_reward_threshold=.2, lr=.001,
                policy_network_path=policy_network_path)
        elif args.rl_method == 'a3c':
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_unit.append(min_trading_unit)
            list_max_trading_unit.append(max_trading_unit)
        
        if learner is not None:
            learner.fit(balance=10000000, num_epoches=100,
                        discount_factor=0.99, start_epsilon=.2)

            # 신경망을 파일로 저장
            learner.save_models()

    if args.rl_method == 'a3c':
        learner = A3CLearner(
            list_stock_code=list_stock_code, list_chart_data=list_chart_data, list_training_data=list_training_data,
            list_min_trading_unit=list_min_trading_unit, list_max_trading_unit=list_max_trading_unit,
            delayed_reward_threshold=.2, lr=.001, policy_network_path=policy_network_path)
        learner.fit(balance=10000000, num_epoches=100,
                    discount_factor=0.99, start_epsilon=.2)