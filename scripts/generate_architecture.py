import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Function to draw box with text
    def draw_box(x, y, width, height, text, color='lightblue'):
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10, wrap=True)
        return x + width/2, y, y + height # return center x, bottom y, top y

    # 1. Input Layer
    bx, by_bottom, by_top = draw_box(1, 8, 3, 1.5, "Student Activity\n(LMS Logs, Videos, Quizzes)", color='#e1f5fe')
    
    # 2. Preprocessing
    px, py_bottom, py_top = draw_box(1, 5, 3, 1.5, "Data Preprocessing\n(Cleaning, SMOTE Balancing)", color='#fff9c4')
    
    # Arrow 1->2
    ax.annotate('', xy=(px, py_top), xytext=(bx, by_bottom),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # 3. Feature Engineering
    fx, fy_bottom, fy_top = draw_box(4.5, 5, 3, 1.5, "Feature Engineering\n(FSLSM Mapping)", color='#ffe0b2')
    
    # Arrow 2->3
    ax.annotate('', xy=(4.5, 5.75), xytext=(4, 5.75),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # 4. AI Models (Core)
    mx, my_bottom, my_top = draw_box(8, 5, 3.5, 3, "AI Core Models\n\n- Hybrid Neural Networks\n- Graph Learning + Fuzzy C-Means\n- Ensemble (LightGBM/CatBoost)", color='#ffccbc')

    # Arrow 3->4
    ax.annotate('', xy=(8, 5.75), xytext=(7.5, 5.75),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # 5. Semi-Supervised Loop
    sx, sy_bottom, sy_top = draw_box(8, 2, 3.5, 1.5, "Semi-Supervised Learning\n(Self-training with unlabeled data)", color='#d1c4e9')

    # Arrows between 4 and 5 (Cyclic)
    ax.annotate('', xy=(9.75, my_bottom), xytext=(9.75, sy_top),
                arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax.annotate('', xy=(8.5, sy_top), xytext=(8.5, my_bottom),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # 6. Output
    ox, oy_bottom, oy_top = draw_box(8, 8, 3.5, 1.5, "Output Dashboard\n(Predicted Learning Style)", color='#c8e6c9')

    # Arrow 4->6
    ax.annotate('', xy=(mx, oy_bottom), xytext=(mx, my_top),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.title("System Architecture: AI-Driven Learning Style Prediction", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("../docs/../docs/architecture_diagram.png", dpi=300)
    print("Architecture diagram saved as ../docs/../docs/architecture_diagram.png")

if __name__ == "__main__":
    create_architecture_diagram()
