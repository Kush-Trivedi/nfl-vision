# Import Libraries
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Constant Graphics Params
plt.rcParams['figure.dpi'] = 180
plt.rcParams["figure.figsize"] = (22,16)
colors=sns.color_palette('Set3')
sns.set(rc={
    'axes.facecolor':'#FFFFFF', 
    'figure.facecolor':'#FFFFFF',
    'font.sans-serif':'Arial',
    'font.family':'sans-serif'
})

class NFLFiled:
    def __init__(self, width, height, color="white"):
        self.width = width
        self.height = height
        self.color = color
        self.fig, self.ax = self.create_pitch()

    def create_pitch(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.axis('off')
        
        background = Rectangle((0, 0), self.width, self.height, linewidth=1, facecolor='#CDB599FF', edgecolor='black', capstyle='round')
        ax.add_patch(background)

        _ = [ax.plot([10 + 5 * i, 10 + 5 * i], [0, self.height], c="black", linestyle='--', lw=1, alpha=0.2) for i in range(21) if i % 2 != 0]
        _ = [ax.plot([10 + 10 * i, 10 + 10 * i], [0, self.height], c="black", lw=1, alpha=0.3) for i in range(11)]

        for units in range(10, 100, 10):
            units_text = units if units <= 100 / 2 else 100 - units 
            ax.text(10 + units - 1.1, self.height - 7.5, units_text, size=15, c="black", weight="bold", alpha=0.2)
            ax.text(10 + units - 1.1, 7.5, units_text, size=15, c="black", weight="bold", rotation=180, alpha=0.2)

        _ = [ax.plot([10 + x * 5 + j, 10 + x * 5 + j], [1, 3], color="black", lw=1, alpha=0.2) for x in range(20) for j in range(1, 5)]
        _ = [ax.plot([10 + x * 5 + j, 10 + x * 5 + j], [self.height - 1, self.height - 3], color="black", lw=1, alpha=0.2) for x in range(20) for j in range(1, 5)]
        y = (self.height - 18.5) / 2
        _ = [ax.plot([10 + x * 5 + j, 10 + x * 5 + j], [y, y + 2], color="black", lw=1, alpha=0.2) for x in range(20) for j in range(1, 5)]
        _ = [ax.plot([10 + x * 5 + j, 10 + x * 5 + j], [self.height - y, self.height - y - 2], color="black", lw=1, alpha=0.2) for x in range(20) for j in range(1, 5)]
        
        ax.text(2.5, (self.height - 10) / 2, "ENDZONE", size=20, c="white", weight="bold", rotation=90)
        end_zone_left = Rectangle((0, 0), 10, self.height, ec=self.color, fc="black", lw=1)
        ax.add_patch(end_zone_left)   
        ax.text(self.width - 7.5, (self.height - 10) / 2, "ENDZONE", size=20, c="white", weight="bold", rotation=-90)
        end_zone_right = Rectangle((self.width - 10, 0), 10, self.height, ec=self.color, fc="black", lw=1)
        ax.add_patch(end_zone_right)
        return fig, ax

    def save_pitch(self, folder_path, filename='pitch.png'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        fill_file_path = os.path.join(folder_path, filename)
        self.fig.savefig(fill_file_path, bbox_inches='tight')
        plt.close(self.fig)

