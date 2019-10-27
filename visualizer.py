import threading
import numpy as np
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from mpl_finance import candlestick_ohlc


lock = threading.Lock()


class Visualizer:

    def __init__(self, vnet=False):
        self.canvas = None
        self.fig = None  # 캔버스 같은 역할을 하는 Matplotlib의 Figure 클래스 객체
        self.axes = None  # 차트를 그리기 위한 Matplotlib의 Axes 클래스 객체
        self.ax_v = None  # 가치 신경망 출력
        self.vnet = vnet  # 가치 신경망 출력 활성화

    def prepare(self, chart_data):
        with lock:
            # 캔버스를 초기화하고 4개의 차트를 그릴 준비
            # self.fig, self.axes = plt.subplots(nrows=4, ncols=1, facecolor='w', sharex=True)
            self.fig = Figure(facecolor='w')
            self.canvas = FigureCanvas(self.fig)
            self.axes = self.fig.subplots(nrows=4, ncols=1, sharex=True)

            self.ax_v = self.axes[2].twinx()
            for ax in self.axes:
                # 보기 어려운 과학적 표기 비활성화
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
            # 차트 1. 일봉 차트
            self.axes[0].set_ylabel('Env.')  # y 축 레이블 표시
            # 거래량 가시화
            x = np.arange(len(chart_data))
            volume = np.array(chart_data)[:, -1].tolist()
            self.axes[0].bar(x, volume, color='b', alpha=0.3)
            # ohlc란 open, high, low, close의 약자로 이 순서로된 2차원 배열
            ax = self.axes[0].twinx()
            ohlc = np.hstack((x.reshape(-1, 1), np.array(chart_data)[:, 1:-1]))
            # self.axes[0]에 봉차트 출력
            # 양봉은 빨간색으로 음봉은 파란색으로 표시
            candlestick_ohlc(ax, ohlc, colorup='r', colordown='b')

    def plot(self, epoch_str=None, num_epoches=None, epsilon=None,
            action_list=None, actions=None, num_stocks=None,
            outvals_policy=None, outvals_value=None, exps=None, learning=None,
            initial_balance=None, pvs=None):
        with lock:
            x = np.arange(len(actions))  # 모든 차트가 공유할 x축 데이터
            actions = np.array(actions)  # 에이전트의 행동 배열
            outvals_policy = np.array(outvals_policy)  # 정책 신경망의 출력 배열
            if outvals_value is not None:
                outvals_value = np.array(outvals_value)  # 가치 신경망의 출력 배열
            pvs_base = np.zeros(len(actions)) + initial_balance  # 초기 자본금 배열

            # 차트 2. 에이전트 상태 (행동, 보유 주식 수)
            colors = ['r', 'b']
            for actiontype, color in zip(action_list, colors):
                for i in x[actions == actiontype]:
                    self.axes[1].axvline(i, color=color, alpha=0.1)  # 배경 색으로 행동 표시
            self.axes[1].plot(x, num_stocks, '-k')  # 보유 주식 수 그리기

            # 차트 3. 신경망의 출력 및 탐험
            for exp_idx in exps:
                # 탐험을 노란색 배경으로 그리기
                self.axes[2].axvline(exp_idx, color='y')
            # 가치 신경망 출력
            if self.vnet:
                color = 'k'
                self.ax_v.plot(x, outvals_value, color=color, linestyle='-')
                self.ax_v.tick_params(axis='y', labelcolor=color)
            for idx, outval in zip(x, outvals_policy):
                color = 'white'
                if outval.argmax() == 0:
                    color = 'r'  # 매수면 빨간색
                elif outval.argmax() == 1:
                    color = 'b'  # 매도면 파란색
                elif outval.argmax() == 2:
                    color = 'g'  # 홀드면 초록색
                # 행동을 배경으로 그리기
                self.axes[2].axvline(idx, color=color, alpha=0.1)
            styles = ['.r', '.b', '.g']
            for action, style in zip(action_list, styles):
                # 정책 신경망의 출력 점으로 그리기
                self.axes[2].plot(x, outvals_policy[:, action], style)

            # 차트 4. 포트폴리오 가치
            self.axes[3].axhline(initial_balance, linestyle='-', color='gray')
            self.axes[3].fill_between(x, pvs, pvs_base,
                                    where=pvs > pvs_base, facecolor='r', alpha=0.1)
            self.axes[3].fill_between(x, pvs, pvs_base,
                                    where=pvs < pvs_base, facecolor='b', alpha=0.1)
            self.axes[3].plot(x, pvs, '-k')
            for learning_idx, delayed_reward in learning:
                # 학습 위치 표시
                if delayed_reward > 0:
                    self.axes[3].axvline(learning_idx, color='r', alpha=0.1)
                else:
                    self.axes[3].axvline(learning_idx, color='b', alpha=0.1)

            # 에포크 및 탐험 비율
            self.fig.suptitle('Epoch %s/%s (e=%.2f)' % (epoch_str, num_epoches, epsilon))
            # 캔버스 레이아웃 조정
            self.fig.tight_layout()
            self.fig.subplots_adjust(top=.9)

    def clear(self, xlim):
        with lock:
            _axes = self.axes.tolist()
            _axes.append(self.ax_v)
            for ax in _axes[1:]:
                ax.cla()  # 그린 차트 지우기
                ax.relim()  # limit를 초기화
                ax.autoscale()  # 스케일 재설정
            # y축 레이블 재설정
            self.axes[1].set_ylabel('Agent')
            self.axes[2].set_ylabel('Net.')
            if self.vnet: self.ax_v.set_ylabel('V')
            self.axes[3].set_ylabel('PV')
            for ax in _axes:
                ax.set_xlim(xlim)  # x축 limit 재설정
                ax.get_xaxis().get_major_formatter().set_scientific(False)  # 과학적 표기 비활성화
                ax.get_yaxis().get_major_formatter().set_scientific(False)  # 과학적 표기 비활성화
                ax.ticklabel_format(useOffset=False)  # x축 간격을 일정하게 설정

    def save(self, path):
        with lock:
            self.fig.savefig(path)
