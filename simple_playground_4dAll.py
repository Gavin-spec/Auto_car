import math as m
import random as r
from simple_geometry import *
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.cluster import KMeans
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class Car():
    def __init__(self) -> None:
        self.radius = 5
        self.angle_min = -90
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5

        self.reset()

    @property
    def diameter(self):
        return self.radius/2

    def reset(self):
        self.angle = 90
        self.wheel_angle = 0

        xini_range = (self.xini_max - self.xini_min - self.radius)
        left_xpos = self.xini_min + self.radius//2
        self.xpos = r.random()*xini_range + left_xpos  # random x pos [-3, 3]
        self.ypos = 0

    def setWheelAngle(self, angle):
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle <= self.wheel_min else self.wheel_max)

    def setPosition(self, newPosition: Point2D):
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    def getPosition(self, point='center') -> Point2D:
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.radius/2, 0).rorate(right_angle)
            return Point2D(self.xpos, self.ypos) + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.radius/2, 0).rorate(left_angle)
            return Point2D(self.xpos, self.ypos) + left_point

        elif point == 'front':
            fx = m.cos(self.angle/180*m.pi)*self.radius/2+self.xpos
            fy = m.sin(self.angle/180*m.pi)*self.radius/2+self.ypos
            return Point2D(fx, fy)
        else:
            return Point2D(self.xpos, self.ypos)

    def getWheelPosPoint(self):
        wx = m.cos((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.xpos
        wy = m.sin((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.ypos
        return Point2D(wx, wy)

    def setAngle(self, new_angle):
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    def tick(self):
        '''
        set the car state from t to t+1
        '''
        car_angle = self.angle/180*m.pi
        wheel_angle = self.wheel_angle/180*m.pi
        new_x = self.xpos + float(m.cos(car_angle+wheel_angle)) + \
            float(m.sin(wheel_angle)*m.sin(car_angle))

        new_y = self.ypos + float(m.sin(car_angle+wheel_angle)) - \
            float(m.sin(wheel_angle)*m.cos(car_angle))
        new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) / (self.radius*1.5))) / m.pi * 180

        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)
        

class RBFN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, n_iters=1000, sigma=1.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.sigma = sigma  # RBF kernel's width parameter
        
        # 初始化权重和偏置
        self.weights = np.random.randn(hidden_size, output_size)
        self.bias = np.zeros(output_size)

    def initialize_centers_with_kmeans(self, X):
        # 使用K-Means找到隐藏层中心
        kmeans = KMeans(n_clusters=self.hidden_size, random_state=0).fit(X)
        self.centers = kmeans.cluster_centers_

    def rbf_kernel(self, X, centers):
        # 高斯核函数计算输入点到各中心点的距离
        return np.exp(-cdist(X, centers, 'sqeuclidean') / (2 * self.sigma ** 2))

    def fit(self, X, y):
        # 使用K-Means初始化中心
        self.initialize_centers_with_kmeans(X)
        
        for epoch in range(self.n_iters):
            # 计算 RBF 层输出（高斯核值）
            hidden_output = self.rbf_kernel(X, self.centers)
            
            # 计算输出层预测值
            output = hidden_output.dot(self.weights) + self.bias
            
            # 计算误差
            error = output - y
            
            # 更新权重和偏置（简单的梯度下降）
            self.weights -= self.learning_rate * hidden_output.T.dot(error) / len(X)
            self.bias -= self.learning_rate * np.sum(error) / len(X)
            
            # 每100次迭代打印一次误差
            if epoch % 100 == 0:
                mse = np.mean(error ** 2)
                print(f"Epoch {epoch}: MSE = {mse:.4f}")

    def predict(self, X):
        # 计算 RBF 层输出（高斯核值）
        hidden_output = self.rbf_kernel(X, self.centers)
        
        # 计算输出层预测值
        return hidden_output.dot(self.weights) + self.bias
    
    def calculate_mse(self, X, y):
        # 计算均方误差
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse

    





class Playground():
    def __init__(self):
        # read path lines
        
        self.path_line_filename = "軌道座標點.txt"
        self._setDefaultLine()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
        ]

        self.car = Car()
        self.train()
        self.reset()

    def _setDefaultLine(self):
        print('use default lines')
        # default lines
        self.destination_line = Line2D(18, 40, 30, 37)

        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None

    def _readPathLines(self):
        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination line
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            self._setDefaultLine()
    def train(self):
        df = pd.read_csv('train4dAll.txt',header=None)[0].str.split(' ', expand=True)
        df = df.astype(float)
        action_train = (df.iloc[:, -1] - self.car.wheel_min)*(self.n_actions-1)/(self.car.wheel_max-self.car.wheel_min) 
        
        X = np.array(df.iloc[:, 0:3])
        y = np.array(action_train).reshape(len(action_train), 1)  

        # 标准化输入特征
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / (X_std)
        self.model = RBFN(3, 100, 1, 0.02,10000)
        self.model.fit( X, y)
        self.X_mean = X_mean
        self.X_std = X_std
        
        # 使用前三列预测第四列（方向盘角度）
        # print(self.model.predict(X))
    def predictAction(self,state,x,y):
        # X = [ [x] + [y] + state ]
        X = [state ]
        X = np.array((X - self.X_mean) / (self.X_std))
        action = self.model.predict(X)[0]
        return  action 

    @property
    def n_actions(self):  # action = [0~num_angles-1]
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):
        return (len(self.state),)

    @ property
    def state(self):
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    def _checkDoneIntersects(self):
        if self.done:
            return self.done

        cpos = self.car.getPosition('center')     # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        diameter = self.car.diameter

        isAtDestination = cpos.isInRect(
            self.destination_line.p1, self.destination_line.p2
        )
        done = False if not isAtDestination else True

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # chack every line in play ground
            dToLine = cpos.distToLine2D(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos-p1).length, (cpos-p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < diameter)
            p2_touch = (dp2 < diameter)
            body_touch = (
                dToLine < diameter and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                # print((dp1 - diameter),(dp2 - diameter),body_touch,front_touch)
                if not done:
                    done = True

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1)*front_u+p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1)*right_u+p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1)*left_u+p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self._setIntersections(front_intersections,
                               left_intersections,
                               right_intersections)
        
        # results
        self.done = done
        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def reset(self):
        self.done = False
        self.car.reset()

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)

        self._checkDoneIntersects()
        return self.state

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)

        self._checkDoneIntersects()

    def calWheelAngleFromAction(self, action):
        angle = self.car.wheel_min + \
            action*(self.car.wheel_max-self.car.wheel_min) / \
            (self.n_actions-1)
        # print(self.car.wheel_max,self.car.wheel_min,angle)
        return angle

    def step(self, action=None):
        '''
        請更改此處code，依照自己的需求撰寫。
        '''
        if action:
            angle = self.calWheelAngleFromAction(action=action)
            self.car.setWheelAngle(angle)

        if not self.done:
            self.car.tick()

            self._checkDoneIntersects()
            return self.state
        else:
            return self.state



class PlaygroundUI(tk.Tk):
    def __init__(self, playground, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Playground Simulation")
        self.geometry("800x600")
        
        # Playground 和車輛的物件
        self.playground = playground
        
        # 初始化 matplotlib 圖表
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 繪製初始的 Playground 狀態
        self.plot_playground()
        
        # 按鈕來觸發車輛的下一步行動
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        self.next_button = ttk.Button(button_frame, text="Next Step", command=self.next_step)
        self.next_button.pack(side=tk.LEFT)
        
    def plot_playground(self):
        self.ax.clear()
        
        # 繪製軌道線和裝飾線
        for line in self.playground.lines + self.playground.decorate_lines:
            self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'k-')
        
        # 繪製終點線
        line = self.playground.destination_line
        self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'g--')
        
        # 繪製車的當前位置
        self.plot_car()
        self.canvas.draw()
        
    def plot_car(self):
        car = self.playground.car
        car_pos = car.getPosition()
        
        # 車輛中心位置和前端方向
        x, y = car_pos.x, car_pos.y
        angle_rad = car.angle / 180 * np.pi
        
        # 繪製車輛的中心和前端方向
        self.ax.plot(x, y, 'bo', label="Car Center")
        front_x = x + m.cos(angle_rad) * car.radius / 2
        front_y = y + m.sin(angle_rad) * car.radius / 2
        self.ax.plot([x, front_x], [y, front_y], 'b-')
        
    def next_step(self):
        # 隨機生成下一個行動
        action = self.playground.predictAction(self.playground.state)
        
        # 執行一步，更新車輛的位置和方向
        self.playground.step(action)
        
        # 重新繪製
        self.plot_playground()

class UI:
    def __init__(self):
        # 初始化 Playground
        self.playground = Playground()
        self.state = self.playground.reset()
        
        self.root = tk.Tk()
        self.root.title("Playground State")
        
        self.left_dist_label = tk.Label(self.root, text="Left Dist: ")
        self.left_dist_label.pack()
        
        self.front_dist_label = tk.Label(self.root, text="Front Dist: ")
        self.front_dist_label.pack()
        
        self.right_dist_label = tk.Label(self.root, text="Right Dist: ")
        self.right_dist_label.pack()
        
        self.x_label = tk.Label(self.root, text="Car X Position: ")
        self.x_label.pack()
        
        self.y_label = tk.Label(self.root, text="Car Y Position: ")
        self.y_label.pack()

        # 初始化 Matplotlib 图形
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        
        # 关闭默认图形窗口
        plt.close(self.fig)  # 关闭默认的matplotlib图形窗口
        # 绘制初始 Playground
        self.init_playground()

        # 初始化軌跡列表
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_plot, = self.ax.plot([], [], 'r-', label="Trajectory")  # 軌跡線

        # 打開文件以寫入
        self.track_file = open('track4D.txt', 'w')  # 'w' 模式表示寫入（會覆蓋現有文件）

        # 使用 Tkinter 的 after 方法启动更新循环
        self.update_playground()

    def init_playground(self):
        p = self.playground
        for line in p.lines + p.decorate_lines:
            self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'k-')
        
        destination_line = p.destination_line
        self.ax.plot([destination_line.p1.x, destination_line.p2.x], 
                     [destination_line.p1.y, destination_line.p2.y], 'g--', label="Destination")
        
        car_pos = p.car.getPosition('center')
        self.car_plot, = self.ax.plot(car_pos.x, car_pos.y, 'bo', label="Car Position")
        self.direction_plot, = self.ax.plot([], [], 'b-', label="Car Direction")
        
        self.ax.legend()
        self.ax.set_xlim(-10, 40)
        self.ax.set_ylim(-10, 60)
        
    def update_labels(self, state, x, y):
        self.left_dist_label.config(text=f"left_dist: {state[0]}")
        self.front_dist_label.config(text=f"front_dist: {state[1]}")
        self.right_dist_label.config(text=f"right_dist: {state[2]}")
        self.x_label.config(text=f"Car X Position: {x}")
        self.y_label.config(text=f"Car Y Position: {y}")

    def update_playground(self):
        p = self.playground
        if not p.done:
            # 获取动作并更新状态
            car_pos = p.car.getPosition('center')
            action = p.predictAction(self.state, car_pos.x, car_pos.y)
            self.state = p.step(action)
            
            # 更新标签
            self.update_labels(self.state, car_pos.x, car_pos.y)
            
            # 更新車的位置和方向
            self.car_plot.set_data([car_pos.x], [car_pos.y])
            angle_rad = p.car.angle / 180 * m.pi
            front_x = car_pos.x + m.cos(angle_rad) * p.car.radius / 2
            front_y = car_pos.y + m.sin(angle_rad) * p.car.radius / 2
            self.direction_plot.set_data([car_pos.x, front_x], [car_pos.y, front_y])

            # 更新軌跡
            self.trajectory_x.append(car_pos.x)
            self.trajectory_y.append(car_pos.y)
            self.trajectory_plot.set_data(self.trajectory_x, self.trajectory_y)

            # 寫入當前的軌跡到 track4D.txt
            # 格式: x, y, left_dist, front_dist, right_dist

            self.track_file.write(f"{self.state[0]}, {self.state[1]}, {self.state[2]},{ p.car.angle }\n")
            
            # 刷新绘图
            self.canvas.draw()
            
            # 继续更新
            self.root.after(100, self.update_playground)  # 每 100 毫秒更新一次

    def run(self):
        self.root.mainloop()

    def __del__(self):
        # 在程序结束时关闭文件
        self.track_file.close()

if __name__ == "__main__":
    app = UI()
    app.run()










