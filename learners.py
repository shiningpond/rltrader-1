#-*- coding:utf-8 -*-
import os
import locale
import logging
import collections
import threading
import time
import numpy as np
import settings
from environment import Environment
from agent import Agent
import networks
from networks import LSTMNetwork
from visualizer import Visualizer

locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')


class PolicyLearner:
    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.05, 
                 net='lstm', n_steps=1, lr=0.01, policy_network_path=None):
        self.stock_code = stock_code  # 종목코드
        self.chart_data = chart_data
        self.environment = Environment(chart_data)  # 환경 객체
        # 에이전트 객체
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        self.training_data = training_data  # 학습 데이터
        self.sample = None
        self.training_data_idx = -1
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        self.visualizer = Visualizer()  # 가시화 모듈
        
        self.net = net
        self.n_steps = n_steps
        self.lr = lr
        self.policy_network = None

        if self.net == 'dnn':
            pass
        elif self.net == 'lstm':
            # 총 자질 벡터 크기 = 학습 데이터의 자질 벡터 크기 + 에이전트 상태 크기
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, n_steps=self.n_steps, lr=self.lr)
        self.policy_network_path = policy_network_path
        if policy_network_path is not None and os.path.exists(policy_network_path):
            self.policy_network.load_model(model_path=policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1

    def update_networks(self, memory, batch_size, discount_factor, delayed_reward):
        # 배치 학습 데이터 생성
        x, y = self.get_batch(memory, batch_size, discount_factor, delayed_reward)
        if len(x) > 0:
            # 정책 신경망 갱신
            loss = self.policy_network.train_on_batch(x, y)
            return loss
        return None

    def visualize(self, epoch_str, num_epoches, epsilon, memory_action, memory_num_stocks, 
                memory_policy, memory_exp_idx, memory_learning_idx, memory_pv, epoch_summary_dir):
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
            action_list=Agent.ACTIONS, actions=memory_action,
            num_stocks=memory_num_stocks, outvals_policy=memory_policy,
            exps=memory_exp_idx, learning=memory_learning_idx,
            initial_balance=self.agent.initial_balance, pvs=memory_pv
        )
        self.visualizer.save(os.path.join(
            epoch_summary_dir, 'epoch_summary_%s_%s.png' % (
                settings.timestr, epoch_str)))

    def fit(
        self, num_epoches=100, max_memory=60, balance=10000000,
        discount_factor=0, start_epsilon=.5, learning=True):
        logging.info("LR: {lr}, DF: {discount_factor}, "
                    "TU: [{min_trading_unit}, {max_trading_unit}], "
                    "DRT: {delayed_reward_threshold}".format(
            lr=self.policy_network.lr,
            discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        ))

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data)

        # 가시화 결과 저장할 폴더 준비
        epoch_summary_dir = os.path.join(
            settings.BASE_DIR, 'epoch_summary/%s/epoch_summary_%s' % (
                self.stock_code, settings.timestr))
        if not os.path.isdir(epoch_summary_dir):
            os.makedirs(epoch_summary_dir)

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):    
            # 에포크 관련 정보 초기화
            loss = 0.
            itr_cnt = 0
            win_cnt = 0
            exploration_cnt = 0
            batch_size = 0
            pos_learning_cnt = 0
            neg_learning_cnt = 0

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.n_steps)

            # 메모리 초기화
            memory_sample = []
            memory_action = []
            memory_reward = []
            memory_policy = []
            memory_pv = []
            memory_num_stocks = []
            memory_exp_idx = []
            memory_learning_idx = []
            
            # 환경, 에이전트, 정책 신경망 초기화
            self.environment.reset()
            self.agent.reset()
            self.policy_network.reset()
            self.reset()

            # 가시화 초기화
            self.visualizer.clear([0, len(self.chart_data)])

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = 0

            while True:
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # n_step만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.n_steps:
                    continue

                # 정책 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = self.agent.decide_action(
                    self.policy_network, list(q_sample), epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                memory_sample.append(list(q_sample))
                memory_action.append(action)
                memory_reward.append(immediate_reward)
                memory_pv.append(self.agent.portfolio_value)
                memory_num_stocks.append(self.agent.num_stocks)
                memory = [(
                    memory_sample[i],
                    memory_action[i],
                    memory_reward[i])
                    for i in list(range(len(memory_action)))[-max_memory:]
                ]
                if exploration:
                    memory_exp_idx.append(itr_cnt)
                    memory_policy.append([np.nan] * Agent.NUM_ACTIONS)
                else:
                    memory_policy.append(self.policy_network.prob)

                # 반복에 대한 정보 갱신
                batch_size += 1
                itr_cnt += 1
                exploration_cnt += 1 if exploration else 0
                win_cnt += 1 if delayed_reward > 0 else 0

                # 학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신
                if delayed_reward == 0 and batch_size >= max_memory:
                    delayed_reward = immediate_reward
                    self.agent.base_portfolio_value = self.agent.portfolio_value
                if learning and delayed_reward != 0:
                    # 배치 학습 데이터 크기
                    batch_size = min(batch_size, max_memory)
                    # 배치 학습 데이터 생성 및 신경망 갱신
                    _loss = self.update_networks(memory, batch_size, discount_factor, delayed_reward)
                    if _loss is not None:
                        loss += abs(_loss)
                        if delayed_reward > 0:
                            pos_learning_cnt += 1
                        else:
                            neg_learning_cnt += 1
                        memory_learning_idx.append([itr_cnt, delayed_reward])
                    batch_size = 0

            # 에포크 관련 정보 가시화
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            self.visualize(epoch_str, num_epoches, epsilon, memory_action, memory_num_stocks, 
                        memory_policy, memory_exp_idx, memory_learning_idx, memory_pv, epoch_summary_dir)

            # 에포크 관련 정보 로그 기록
            if pos_learning_cnt + neg_learning_cnt > 0:
                loss /= pos_learning_cnt + neg_learning_cnt
            logging.info("[Epoch %s/%s]\tEpsilon:%.4f\t#Expl.:%d/%d\t"
                        "#Buy:%d\t#Sell:%d\t#Hold:%d\t"
                        "#Stocks:%d\tPV:%s\t"
                        "POS:%s\tNEG:%s\tLoss:%10.6f" % (
                            epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
                            self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                            self.agent.num_stocks,
                            locale.currency(self.agent.portfolio_value, grouping=True),
                            pos_learning_cnt, neg_learning_cnt, loss))

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 학습 관련 정보 로그 기록
        logging.info("Max PV: %s, \t # Win: %d" % (
            locale.currency(max_portfolio_value, grouping=True), epoch_win_cnt))

    def get_batch(self, memory, batch_size, discount_factor, delayed_reward):
        x = np.zeros((batch_size, self.n_steps, self.num_features))
        y = np.full((batch_size, self.agent.NUM_ACTIONS), 0)
        for i, (sample, action, reward) in enumerate(memory[-batch_size:]):
            x[i] = np.array(sample).reshape((-1, self.n_steps, self.num_features))
            y[i, action] = delayed_reward
            if discount_factor > 0:
                y[i, action] *= discount_factor ** i
        return x, y

    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    def trade(self, model_path=None, balance=2000000):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)
        self.fit(balance=balance, num_epoches=1, learning=False)

    def save_models(self):
        self.policy_network.save_model(self.policy_network_path)


class A2CLearner(PolicyLearner):
    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05, lr=0.01, policy_network_path=None, 
                 net='lstm', n_steps=1, value_network_path=None, shared_network=None,
                 policy_network=None, value_network=None):
        super().__init__(stock_code, chart_data, training_data=training_data,
                 min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit,
                 delayed_reward_threshold=delayed_reward_threshold, lr=lr, policy_network_path=policy_network_path,
                 net=net,n_steps=n_steps)
        if shared_network is None:
            self.shared_network = networks.get_shared_network(net=net, n_steps=n_steps, input_dim=self.num_features)
        else:
            self.shared_network = shared_network
        if net == 'dnn':
            pass
        elif net == 'lstm':
            if policy_network is None:
                self.policy_network = LSTMNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr, shared_net=self.shared_network)
            else:
                self.policy_network = policy_network
            if value_network is None:
                self.value_network = LSTMNetwork(input_dim=self.num_features, output_dim=1, lr=lr, shared_net=self.shared_network, activation='linear')
            else:
                self.value_network = value_network
        # 모델 로드
        self.policy_network_path = policy_network_path
        self.value_network_path = value_network_path
        if policy_network_path is not None and os.path.exists(policy_network_path):
            self.policy_network.load_model(model_path=policy_network_path)
        if value_network_path is not None and os.path.exists(value_network_path):
            self.value_network.load_model(model_path=value_network_path)
        # 가시화 모듈
        self.visualizer = Visualizer(vnet=True)
        # 가치 신경망 예측값 메모리
        self.memory_value = []

    def update_networks(self, memory, batch_size, discount_factor, delayed_reward):
        # 배치 학습 데이터 생성
        x, y_policy, y_value = self.get_batch(memory, batch_size, discount_factor, delayed_reward)
        if len(x) > 0:
            # 정책 신경망 갱신
            loss_policy = self.policy_network.train_on_batch(x, y_policy)
            loss_value = self.value_network.train_on_batch(x, y_value)
            return loss_policy + loss_value
        return None

    def reset(self):
        super().reset()
        self.memory_value = []

    def build_sample(self):
        sample = super().build_sample()
        if sample is not None:
            self.memory_value.append(self.value_network.predict(sample))
        return sample

    def get_batch(self, memory, batch_size, discount_factor, delayed_reward):
        x = np.zeros((batch_size, 1, self.num_features))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), 0)
        y_value = np.full(batch_size, 0)

        for i, (sample, action, reward) in enumerate(memory[-batch_size:]):
            r = delayed_reward
            adv = delayed_reward - self.memory_value[-batch_size:][i]
            x[i] = np.array(sample).reshape((-1, 1, self.num_features))
            y_policy[i, action] = adv
            y_value[i] = r
            if discount_factor > 0:
                y_policy[i, action] *= discount_factor ** i
                y_value[i] *= discount_factor ** i
        return x, y_policy, y_value

    def visualize(self, epoch_str, num_epoches, epsilon, memory_action, memory_num_stocks, 
                memory_policy, memory_exp_idx, memory_learning_idx, memory_pv, epoch_summary_dir):
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
            action_list=Agent.ACTIONS, actions=memory_action,
            num_stocks=memory_num_stocks, outvals_policy=memory_policy,
            outvals_value=self.memory_value,
            exps=memory_exp_idx, learning=memory_learning_idx,
            initial_balance=self.agent.initial_balance, pvs=memory_pv
        )
        self.visualizer.save(os.path.join(
            epoch_summary_dir, 'epoch_summary_%s_%s.png' % (
                settings.timestr, epoch_str)))

    def save_models(self):
        super().save_models()
        self.value_network.save_model(self.value_network_path)


class A3CLearner:
    def __init__(self, list_stock_code, list_chart_data, list_training_data=None,
                 list_min_trading_unit=None, list_max_trading_unit=None,
                 delayed_reward_threshold=.02, lr=0.01, policy_network_path=None, 
                 net='lstm', n_steps=1, value_network_path=None, shared_network=None):
        if len(list_training_data) == 0: return
        self.num_features = list_training_data[0].shape[1] + Agent.STATE_DIM
        if shared_network is None:
            self.shared_network = networks.get_shared_network(net=net, n_steps=n_steps, input_dim=self.num_features)
        else:
            self.shared_network = shared_network
        if net == 'dnn':
            pass
        elif net == 'lstm':
            self.policy_network = LSTMNetwork(input_dim=self.num_features, output_dim=Agent.NUM_ACTIONS, lr=lr, shared_net=self.shared_network)
            self.value_network = LSTMNetwork(input_dim=self.num_features, output_dim=1, lr=lr, shared_net=self.shared_network, activation='linear')
        # 모델 로드
        self.policy_network_path = policy_network_path
        self.value_network_path = value_network_path
        if policy_network_path is not None and os.path.exists(policy_network_path):
            self.policy_network.load_model(model_path=policy_network_path)
        if value_network_path is not None and os.path.exists(value_network_path):
            self.value_network.load_model(model_path=value_network_path)

        # A2CLearner 생성
        self.learners = []
        for stock_code, chart_data, training_data, min_trading_unit, max_trading_unit in zip(
                list_stock_code, list_chart_data, list_training_data,
                list_min_trading_unit, list_max_trading_unit
            ):
            learner = A2CLearner(stock_code, chart_data, training_data=training_data,
                    min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit,
                    delayed_reward_threshold=delayed_reward_threshold, lr=lr, policy_network_path=policy_network_path, 
                    net=net, n_steps=n_steps, value_network_path=value_network_path, shared_network=self.shared_network,
                    policy_network=self.policy_network, value_network=self.value_network)
            self.learners.append(learner)

    def fit(
        self, num_epoches=100, max_memory=60, balance=10000000,
        discount_factor=0, start_epsilon=.5, learning=True):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(target=learner.fit, daemon=True, kwargs={
                'num_epoches': num_epoches, 'max_memory': max_memory, 'balance': balance,
                'discount_factor': discount_factor, 'start_epsilon': start_epsilon,
                'learning': learning
            }))
        for thread in threads:
            thread.start()
            time.sleep(1)
        for thread in threads: thread.join()
