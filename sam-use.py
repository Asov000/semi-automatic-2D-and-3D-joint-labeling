import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r'/')
from segment_anything import sam_model_registry, SamPredictor


sam_image = cv2.imread(r"C:\Users\25918\Desktop\test2.jpg")
sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)

# 模型路径
sam_checkpoint = r'C:\Users\25918\Downloads\SAM\sam_vit_h_4b8939.pth'

# 模型类型
sam_model_type = 'vit_h'

# 设备类型：cpu / cuda
sam_device = 'cpu'

# 多个提示点，格式是 [x, y]
# 1 表示前景点，0 表示背景点
sam_points = np.array([
    [480, 360],   # 前景点
    [300, 300],   # 背景点
])

sam_labels = np.array([
    1,
    0,
])


class SamPredict:
    def __init__(
        self,
        image: any,
        checkpoint: any,
        model_type: str,
        model_device: str,
        points: np.ndarray,
        labels: np.ndarray,
        random_color: bool
    ):
        self.image = image
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.model_device = model_device
        self.points = points
        self.labels = labels
        self.marker_size = 300
        self.random_color = random_color

    # 显示预标记点
    def show_pre(self) -> None:
        plt.imshow(self.image)
        self.__show_points(self.points, self.labels)
        plt.axis('off')
        plt.show()

    # 显示正负点
    def __show_points(self, points: np.ndarray, labels: np.ndarray) -> None:
        pos_points = points[labels == 1]
        neg_points = points[labels == 0]

        if len(pos_points) > 0:
            plt.scatter(
                pos_points[:, 0],
                pos_points[:, 1],
                color='green',
                marker='*',
                s=self.marker_size,
                edgecolor='white',
                linewidth=1.25
            )

        if len(neg_points) > 0:
            plt.scatter(
                neg_points[:, 0],
                neg_points[:, 1],
                color='red',
                marker='*',
                s=self.marker_size,
                edgecolor='white',
                linewidth=1.25
            )

    # 显示 mask 覆盖
    def __show_mask(self, mask: np.ndarray) -> None:
        if self.random_color:
            color = np.concatenate(
                [np.random.random(3), np.array([0.6])],
                axis=0
            )
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape((1, 1, -1))
        plt.imshow(mask_image)

    # SAM 预测
    def get_result(self) -> None:
        # 加载模型
        model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        model.to(device=self.model_device)

        predictor = SamPredictor(model)

        # 传入图片
        predictor.set_image(self.image)

        # 多点预测
        masks, scores, logits = predictor.predict(
            point_coords=self.points,
            point_labels=self.labels,
            multimask_output=True,
        )

        print('原始图片高度 Height 为:', masks.shape[1])
        print('原始图片宽度 Width 为:', masks.shape[2])
        print('识别主体 Mask 次数为:', masks.shape[0])

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.imshow(self.image)

            # 显示 mask
            self.__show_mask(mask)

            # 显示多个点
            self.__show_points(points=self.points, labels=self.labels)

            plt.title(f"Mask_Times:{i + 1}, Mask_Scores:{score:.4f}", fontsize=18)
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    model_one = SamPredict(
        image=sam_image,
        checkpoint=sam_checkpoint,
        model_type=sam_model_type,
        model_device=sam_device,
        points=sam_points,
        labels=sam_labels,
        random_color=False
    )i


    # 再进行 SAM 分割
    model_one.get_result()
