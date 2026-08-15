import torch
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
import os

def unfold(X):
    # 按阶展开X：输入张量X，输出列表Xlist，其中第n个元素是X按照n阶展开。
    Xdims = list(X.shape)
    # 所有维数放进一个N元列表
    N = len(Xdims)
    # 总维度数/阶数/因子个数
    Xlist = []
    for n in range(N):
        dims = list(range(N))
        dims.remove(n)
        Xn = X.permute([n]+dims).reshape(Xdims[n], -1)

        Xlist.append(Xn)

    return Xlist, Xdims, N

def initAlambdaslist(Xdims, N, R, Xnorm):
    # 初始化因子矩阵：输入X按维展开的Xlist，读取维度，输出n*R的初始化矩阵Alist（列表）
    # 所有An归一化为X范式的（1/n）次方
    Alist = []
    for n in range(N):
        # A = torch.randn(Xdims[n], R, requires_grad=True)
        A = torch.randn(Xdims[n], R)
        Anorm_1 = compute_error(A, 0)
        print(f'A初始化时的范数', Anorm_1)
        A = A * (Xnorm ** (1/N) / Anorm_1)
        Anorm_2 = compute_error(A, 0)
        print(f'A归一化后的范数', Anorm_2)
        A.requires_grad_(True)
        Alist.append(A)

    return Alist


def Hadamard(Alist, N, R):
    # 同一个维度循环单元n中调用
    # 通过循环实现AT*A的hadamard积，输出矩阵V
    V = torch.ones(R, R)
    for k in range(N-1):
        V = V * (Alist[k].T @ Alist[k])

    return V

# 求出A^(-n)，用kronecker积实现
def AnKronecker(Alist, R):
    A_negative_n_list = []
    for r in range(R):
        an_r = torch.ones(1)
        for n in range(len(Alist)):
            an_r = torch.kron(an_r, Alist[n][:, r ])
        A_negative_n_list.append(an_r)
    A_negative_n = torch.stack(A_negative_n_list).T

    return A_negative_n

# 由因子矩阵Alist计算逼近张量Xapprox，任意阶。用双重循环实现。
# def reconstruct(Alist, Xdims, R, N):
#     # 一次性代码，不用放进迭代循环
#     # 由不同维度的展开得到逼近矩阵，输出逼近矩阵Xapprox
#     Xapprox = torch.zeros(Xdims)
#     for r in range(R):
#         Xn = torch.ones(1)
#         for n in range(N):
#             Xn = Xn.unsqueeze(-1) * Alist[n][:, r]
#         Xapprox = Xapprox + Xn.squeeze(0)
#
#     return Xapprox


# # 3阶情形，用Einstein求和约定计算逼近张量。为了代码和任意阶保持一致，传进了3个并不需要的参数。
# def reconstruct(Alist, Xdims, R, N):
#     Xapprox = torch.einsum('ir,jr,kr->ijk', Alist[0], Alist[1], Alist[2])
#
#     return Xapprox

# 4阶情形，用Einstein求和约定计算逼近张量。为了代码省事，传进了3个并不需要的参数
def reconstruct(Alist, Xdims, R, N):
    Xapprox = torch.einsum('ir,jr,kr,lr->ijkl', Alist[0], Alist[1], Alist[2], Alist[3])
    Xapprox_norm = compute_error(Xapprox, 0)
    print(f'Xapprox的F范数是：', Xapprox_norm)

    return Xapprox

def compute_error(X, Y):
    # 计算2个张量的F范数距离/误差
    error = torch.sqrt(torch.sum((X - Y)**2))

    return error

#  对Alist中的元素逐个计算梯度（实际上这只是一个操作，并不需要return，因为梯度是直接存在Alist.grad中的）
def compute_Alist_grad(Alist, Xlist, N, R):
    for n in range(N):
        Xn = Xlist[n]
        An = Alist[n]
        A_remove_n_list = Alist[:n] + Alist[n + 1:]
        A_negative_n = AnKronecker(A_remove_n_list, R)
        Gamman = Hadamard(A_remove_n_list, N, R)

        gradAn = (-Xn) @ A_negative_n + An @ Gamman

        Alist[n].grad = gradAn

    return Alist


if __name__ == "__main__":

    torch.manual_seed(42) # 生成输入矩阵的随机种子固定
    grad_mode = 'manual' # 'auto'/ 'manual' #是否自动求导，用于对比求导情况
    opt_method = 'adam' # 优化器选择'adam' / 'sgd' / 'gd_manual' / 'lbfgs'
    iternum = 800 # 迭代次数
    LearningRate = 0.05 # adam学习率
    LearningRate_gdmanual = 1e-3 # 注意，sgd和gd_manual的学习率都必须非常小，adam的学习率可以比较大
    R = 300 # 低秩逼近真实视频数据时，逼近张量的秩的大小

    # # （1）随机生成特定秩张量
    # I, J, K, True_R = 10, 10, 10, 5
    # A_true = torch.randn(I, True_R)
    # B_true = torch.randn(J, True_R)
    # C_true = torch.randn(K, True_R)
    # # 用爱因斯坦求和约定快速构造张量 X (相当于把三个矩阵按列做外积并相加)
    # X = torch.einsum('ir,jr,kr->ijk', A_true, B_true, C_true)
    # R = True_R  # 合成张量的秩分解实验，逼近张量的秩直接与合成张量相等，用于验证算法的准确性

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
    print(f'一开始输入的时候的范数',Xnorm)
    # 维度转换
    X = X.permute(0, 1, 3, 2)

    # （2）优化张量分解初始化随机种子不固定，使重复实验有意义
    torch.seed()

    print('开始张量分解...')
    # 开始计时
    start_time = time.time()
    total_grad_time = 0.0

    # （3）按维展开输入张量X
    Xlist, Xdims, N = unfold(X)

    # （4）初始化因子矩阵
    Alist = initAlambdaslist(Xdims, N, R, Xnorm)

    # （5）计算梯度并更新参数
    # 选择优化方法
    if opt_method == 'adam':
        # （5.1）Adam优化器
        optimizer = torch.optim.Adam(Alist, lr=LearningRate)
        # （5.2）自动更新学习率，每过500轮更新学习率一次
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    if opt_method == 'sgd':
        # （5.3）SGD优化器
        optimizer = torch.optim.SGD(Alist, lr=LearningRate_gdmanual, momentum=0.9)
    if opt_method == 'lbfgs':
        # （5.4）LBFGS优化器
        optimizer = torch.optim.LBFGS(Alist, lr=1.0, max_iter=20, max_eval=25, history_size=10, line_search_fn='strong_wolfe')
        iternum = 20  # lbfgs单列迭代次数

    # （5.5）计算梯度
    for epoch in range(iternum):
        if opt_method == 'adam' or opt_method == 'sgd':
            optimizer.zero_grad()

        # （5.5.1）手动求导，复现论文
        if grad_mode == 'manual' and opt_method != 'lbfgs':
            with torch.no_grad():
                Xapprox = reconstruct(Alist, Xdims, R, N)
                approx_error = compute_error(X, Xapprox)

                # 计时点
                t0 = time.time()
                # 计算梯度
                compute_Alist_grad(Alist, Xlist, N, R)

                t1 = time.time()
                total_grad_time += (t1 - t0)

        # （5.5.2）自动求导
        elif grad_mode == 'auto'and opt_method != 'lbfgs':
            Xapprox = reconstruct(Alist, Xdims, R, N)
            loss = 0.5 * torch.sum((Xapprox - X) ** 2)

            t0 = time.time()
            # 反向传播自动计算梯度
            loss.backward()

            t1 = time.time()
            total_grad_time += (t1 - t0)

            with torch.no_grad():
                approx_error = compute_error(X, Xapprox)

        if opt_method == 'adam':
            optimizer.step()  # 更新参数

            scheduler.step()  # 调整学习率

        if opt_method == 'sgd':

            optimizer.step()  # 更新参数

        if opt_method == 'lbfgs':
            def closure():
                global total_grad_time
                optimizer.zero_grad()
                Xapprox = reconstruct(Alist, Xdims, R, N)
                loss = 0.5 * torch.sum((Xapprox - X) ** 2)
                if grad_mode == 'auto':
                    t0 = time.time()
                    loss.backward()
                    t1 = time.time()
                    total_grad_time += (t1 - t0)
                elif grad_mode == 'manual':
                    with torch.no_grad():
                        t0 = time.time()
                        compute_Alist_grad(Alist, Xlist, N, R)
                        t1 = time.time()
                        total_grad_time += (t1 - t0)
                return loss
            loss = optimizer.step(closure)
            with torch.no_grad():
                approx_error = torch.sqrt(loss * 2)
                print(f'当前的逼近误差是：', approx_error)
            Xapprox = reconstruct(Alist, Xdims, R, N)


        if opt_method == 'gd_manual':
            with torch.no_grad():
                for n in range(N):
                    Alist[n] -= LearningRate_gdmanual * Alist[n].grad
                    Alist[n].grad.zero_()

        if epoch % 100 == 0 and opt_method != 'lbfgs':
            print(f"Epoch {epoch} | loss: {approx_error.item():.6f}")
        if epoch % 5 == 0 and opt_method == 'lbfgs':
            print(f"Epoch {epoch} | Loss: {approx_error.item():.6f}")

    end_time = time.time()
    print(f'张量分解完成，用时{end_time - start_time}秒')
    print(f'梯度计算总用时{total_grad_time}秒')


    # （6）计算逼近张量的F范数
    Xnorm = compute_error(X, 0)
    Xapproxnorm = compute_error(Xapprox, 0)



    print(f'迭代次数是', iternum)
    # print(f'输入张量是：',X)
    print(f'输入张量的大小是：', X.shape)
    # print(f'因子矩阵是：', Alist)
    # print(f'逼近张量是：', Xapprox)
    # print(f'误差是', approx_error)
    print(f'输入张量的F范数是', Xnorm)
    # print(f'逼近张量的F范数是', Xapproxnorm)
    print(f'相对误差是', approx_error / Xnorm)







    save_dir = r"D:\LearningFiles\UndergraduateThesis"
    # 如果文件夹不存在，自动创建
    os.makedirs(save_dir, exist_ok=True)

    # 动态生成文件名
    img_filename = f"adam_SinglePicture_Recon_akiyo_picture_R={R}_iter={iternum}.png"
    video_filename = f"adam_Recon_akiyo_video_R={R}_iter={iternum}.mp4"

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

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Video (Frame 150)")
    # plt.imshow(img_true)
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    plt.title(f"CP Reconstructed (R={R})")
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

