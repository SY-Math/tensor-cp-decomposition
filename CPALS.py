import torch
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
import os



def unfold(X):
    # 一次性代码
    # 输入X张量，输出列表，第n元素是X按照n维展开。从这里也能看出按n维展开，其余元素是按照阶数从小到达往上排的。
    Xdims = list(X.shape)
    # 所有维度放成一个n维列表
    N = len(Xdims)
    # 总维度数/阶数/因子个数
    Xlist = []
    for n in range(N):
        dims = list(range(N))
        dims.remove(n)
        Xn = X.permute([n]+dims).reshape(Xdims[n], -1)
        Xlist.append(Xn)
        # 注意：permute是按照某个顺序（以列表为参数）对张量维度重排
        # reshape才是变换形状

    return Xlist, Xdims, N

def initAlambdaslist(Xdims, N, R):
    # 一次性代码
    # 输入X按维展开的Xlist，读取维度，输出n*R的初始化矩阵Alist（列表），范数列表lambdaslist
    Alist = []
    lambdaslist = []
    for n in range(N):
        A = torch.randn(Xdims[n], R)
        lambdas = torch.norm(A, dim=0)
        A = A / lambdas
        Alist.append(A)
        lambdaslist.append(lambdas)

    return Alist, lambdaslist

def Hadamard(Alist, N, R):
    # 同一个维度循环单元n中调用
    # 通过循环实现AT*A的hadamard积，输出矩阵V
    V = torch.ones(R, R)
    for k in range(N-1):
        V = V * (Alist[k].T @ Alist[k])

    return V

def pseudoinverse(V):
    # 同一个维度循环单元n中调用
    # 求解伪逆，输出Vinverse
    Vinverse = torch.linalg.pinv(V)

    return Vinverse

def KhatriRao(Alist, N, R):
    # 同一个维度循环单元n中调用
    Akr_cols = []
    for r in range(R):
        clo = torch.ones(1)
        for n in range(N-1):
            clo = torch.kron(clo, Alist[n][:, r])
        Akr_cols.append(clo.unsqueeze(1))
    Akr = torch.cat(Akr_cols, dim=1)

    return Akr

def interA(Xlist, Akr, Vinverse, lambdaslist, n):
    # 同一个维度循环单元n中调用
    # 计算An迭代结果，输出更新后的列表Alist，lambdaslist
    An = Xlist[n] @ Akr @ Vinverse
    lambdas = torch.norm(An, dim=0)
    An = An / lambdas
    lambdaslist[n] = lambdas

    return An, lambdaslist

def reconstruct(Alist, lambdaslist, Xdims, R, N):
    # 一次性代码，不用放进迭代循环
    # 由不同维度的展开得到逼近矩阵，输出逼近矩阵Xapprox
    Xapprox = torch.zeros(Xdims)
    weights = lambdaslist[-1]
    for r in range(R):
        Xn = torch.ones(1) * weights[r]
        for n in range(N):
            Xn = Xn.unsqueeze(-1) * Alist[n][:, r]
        Xapprox = Xapprox + Xn.squeeze(0)

    return Xapprox

def compute_error(X, Y):
    # 计算2个张量的F范数距离/误差
    error = torch.sqrt(torch.sum((X - Y)**2))

    return error

if __name__ == "__main__":

    torch.manual_seed(42)

    iternum = 200
    R = 100  # 低秩逼近真实视频数据时，逼近张量的秩的大小

    # # （1）随机生成特定秩张量
    # I, J, K, True_R = 100, 100, 100, 100
    # A_true = torch.randn(I, True_R)
    # B_true = torch.randn(J, True_R)
    # C_true = torch.randn(K, True_R)
    # # 用爱因斯坦求和约定快速构造张量 X (相当于把三个矩阵按列做外积并相加)
    # X = torch.einsum('ir,jr,kr->ijk', A_true, B_true, C_true)
    # R = True_R
    #
    # print(f'输入张量是：',X)
    # print(f'输入张量的大小是：', X.shape)

    # （1.5）真实视频数据读取
    file_path = r"D:\LearningFiles\UndergraduateThesis\176x144\akiyo.mat"
    with h5py.File(file_path, 'r') as f:
        numpy_data = np.array(f['X'])
    print(f'输入数据的形状是', numpy_data.shape)
    print(f'输入数据的最大值是', numpy_data.max())
    print(f'输入数据的最小值是', numpy_data.min())
    print(f'输入数据的平均值是', np.mean(numpy_data))

    # 输入数据转换成张量
    X = torch.tensor(numpy_data, dtype=torch.float32)
    Xnorm = compute_error(X, 0)
    print(f'一开始输入的时候的范数', Xnorm)
    # 维度转换
    X = X.permute(0, 1, 3, 2)

    torch.seed()

    Xlist, Xdims, N = unfold(X)

    start_time = time.time()

    Alist, lambdaslist = initAlambdaslist(Xdims, N, R)

    for k in range(iternum):
        for n in range(N):
            Alist = Alist[:n] + Alist[n+1:]
            V = Hadamard(Alist, N, R)
            Vinverse = pseudoinverse(V)
            Akr = KhatriRao(Alist, N, R)
            An, lambdaslist = interA(Xlist, Akr, Vinverse, lambdaslist, n)
            Alist = Alist[:n] + [An] + Alist[n:]
            print(f'现在是第', k, '次迭代，第', n, '个维度')
        Xapprox = reconstruct(Alist, lambdaslist, Xdims, R, N)
        error = compute_error(X, Xapprox)
        print(f'现在是第', k, '次迭代，误差是', error)


    end_time = time.time()
    total_time = end_time - start_time
    print(f'总时间是', total_time)


    Xapprox = reconstruct(Alist, lambdaslist, Xdims, R, N)
    approx_error = compute_error(X, Xapprox)
    Xnorm = compute_error(X, 0)
    Xapproxnorm = compute_error(Xapprox, 0)

    # print(f'因子矩阵是：', Alist)
    # print(f'逼近张量是：', Xapprox)
    print(f'误差是', approx_error)
    print(f'输入张量的F范数是', Xnorm)
    print(f'逼近张量的F范数是', Xapproxnorm)
    print(f'相对误差是', approx_error / Xnorm)





    # 可视化
    save_dir = r"D:\LearningFiles\UndergraduateThesis"
    # 如果文件夹不存在，自动创建
    os.makedirs(save_dir, exist_ok=True)

    # 动态生成文件名
    img_filename = f"ALSCP_Recon_akiyo_picture_R={R}_iter={iternum}_补测时间.png"
    video_filename = f"ALSCP_Recon_akiyo_video_R={R}_iter={iternum}_补测时间.mp4"

    img_path = os.path.join(save_dir, img_filename)
    video_path = os.path.join(save_dir, video_filename)

    # ================= 2. 生成并保存静态对比图 =================
    print("正在生成静态对比图...")
    frame_idx = 150

    # 转换为 numpy，调整通道顺序为 [H, W, C]
    img_true = X[frame_idx].permute(1, 2, 0).numpy()
    img_approx = Xapprox[frame_idx].detach().permute(1, 2, 0).numpy()

    # 数据本身就是 0~1 的浮点数，直接限制在 0.0~1.0 之间防止越界报错即可
    img_true = np.clip(img_true, 0.0, 1.0)
    img_approx = np.clip(img_approx, 0.0, 1.0)

    plt.figure(figsize=(5, 5))

    # # 如果需要2张图片显示，则需要下面的段落（包含调整大小）
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Video (Frame 150)")
    # plt.imshow(img_true)
    # plt.axis('off')

    # plt.subplot(1, 2, 2) # 这是第二张图分块用的
    plt.title(f"ALSCP Reconstructed (R={R})")
    plt.imshow(img_approx)
    plt.axis('off')

    plt.tight_layout()

    # 在 plt.show() 之前保存图片，dpi=300 保证论文打印清晰
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"图片已成功保存至: {img_path}")

    plt.show()

    # ================= 3. 播放并保存对比视频 =================
    print("正在准备播放并保存对比视频...")
    video_true = X.permute(0, 2, 3, 1).numpy()
    video_approx = Xapprox.detach().permute(0, 2, 3, 1).numpy()

    # 因为原始数据是 0~1，为了给 cv2 播放和保存，必须乘上 255 并转为 uint8
    video_true = np.clip(video_true * 255.0, 0, 255).astype(np.uint8)
    video_approx = np.clip(video_approx * 255.0, 0, 255).astype(np.uint8)

    # 【新增】：初始化视频写入器 (VideoWriter)
    # 先拿第一帧计算一下最终画面的宽高
    temp_frame_t = cv2.cvtColor(video_true[0], cv2.COLOR_RGB2BGR)
    temp_frame_a = cv2.cvtColor(video_approx[0], cv2.COLOR_RGB2BGR)
    temp_combined = np.hstack((temp_frame_t, temp_frame_a))
    h, w = temp_combined.shape[:2]
    out_size = (w * 2, h * 2)  # 最终放大了2倍的尺寸 (宽, 高)

    # 使用 mp4v 编码器保存为 .mp4 格式，帧率设为 30 fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_path, fourcc, 30.0, out_size)

    print("按键盘上的 'q' 键可以提前退出播放。")
    for i in range(300):
        frame_t = cv2.cvtColor(video_true[i], cv2.COLOR_RGB2BGR)
        frame_a = cv2.cvtColor(video_approx[i], cv2.COLOR_RGB2BGR)

        combined_frame = np.hstack((frame_t, frame_a))
        height, width = combined_frame.shape[:2]
        combined_frame = cv2.resize(combined_frame, (width * 2, height * 2))

        # 【新增】：将当前帧写入视频文件
        out_video.write(combined_frame)

        cv2.imshow(f'Left: Original | Right: Reconstructed (R={R})', combined_frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    # 【新增】：释放视频写入器，完成保存
    out_video.release()
    cv2.destroyAllWindows()
    print(f"视频已成功保存至: {video_path}")