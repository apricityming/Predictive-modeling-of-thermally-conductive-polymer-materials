import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

savepath = './Figure/'

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(file, columns):
    Thermal_data = load_data(file)
    plt.figure(figsize=(16, 8))

    for i, column in enumerate(columns):
        plt.subplot(2, 3, i + 1)
        sns.distplot(Thermal_data[column], bins=100, kde=True)
        plt.title(f"Distribution of {column}")

    plt.tight_layout()
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format='png', dpi=600, bbox_inches='tight')
    png_buffer.seek(0)
    img = Image.open(png_buffer)
    img.save(savepath+'Feature_distributions.tiff',format='TIFF')

def plot_correlation_heatmap(file):
    Thermal_data = load_data(file)
    plt.rcParams['axes.linewidth'] = 2.5 
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Pearson Correlation Coefficient Heatmap")
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format='png', dpi=600, bbox_inches='tight')
    png_buffer.seek(0)
    img = Image.open(png_buffer)
    img.save(savepath+'correlation_heatmap.tiff',format='TIFF')

def shap_summary_plot(X_train, y_train, X_test):
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    xgb_reg = xgb_reg
    xgb_reg.fit(X_train, y_train)
    explainer = shap.TreeExplainer(xgb_reg, X_train_df.values, feature_names=X_train_df.columns)
    shap_values = explainer.shap_values(X_test_df)
    shap.summary_plot(shap_values, X_test_df, plot_type="bar")
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format='png', dpi=600, bbox_inches='tight')
    png_buffer.seek(0)
    img = Image.open(png_buffer)
    img.save(savepath+'SHAP_summary_plot.tiff',format='TIFF')

    shap.summary_plot2(shap_values2, X_test_df, plot_type="bar")
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format='png', dpi=600, bbox_inches='tight')
    png_buffer.seek(0)
    img = Image.open(png_buffer)
    img.save(savepath+'SHAP_summary_plot.tiff',format='TIFF')

    #SHAP dot plot
    shap.summary_plot(shap_values, X_test_df, plot_type="dot")
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format='png', dpi=600, bbox_inches='tight')
    png_buffer.seek(0)
    img = Image.open(png_buffer)
    img.save(savepath+'SHAP_dot_plot.tiff',format='TIFF')

    #SHAP heatmap plot
    shap.plots.heatmap(shap_values2, max_display = 12)
    plt.savefig(png_buffer, format='png', dpi=600, bbox_inches='tight')
    png_buffer.seek(0)
    img = Image.open(png_buffer)
    img.save(savepath+'SHAP_heatmap_plot.tiff',format='TIFF')

def plot_actual_vs_predicted(y_train, train_predict, y_test, test_predict,savename):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, test_predict, alpha=0.5, marker='s', c=[(22/255, 6/255, 138/255)], label='Test Data')  
    plt.scatter(y_train, train_predict, alpha=0.5, marker='^', c=[(158/255, 24/255, 157/255)], label='Train Data')  
    plt.xlabel("Actual Values (W，m-1，K-1)", fontweight='bold')  
    plt.ylabel("Predicted Values (W，m-1，K-1)", fontweight='bold')  
    plt.title("Actual vs. Predicted Plot", fontweight='bold')  
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], linestyle='--', color=(236/255, 120/255, 83/255), linewidth=2.5)
    plt.plot([min(y_train), max(y_train)], [1.2 * min(y_train), 1.2 * max(y_train)], linestyle='--', color=(253/255, 179/255, 46/255), label='Actual +20%', linewidth=2.5)
    plt.plot([min(y_train), max(y_train)], [0.8 * min(y_train), 0.8 * max(y_train)], linestyle='--', color=(253/255, 179/255, 46/255), label='Actual -20%', linewidth=2.5)
    plt.legend()
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format='png', dpi=600, bbox_inches='tight')
    png_buffer.seek(0)
    img = Image.open(png_buffer)
    img.save(savepath+savename, format='TIFF')
    img.show()

def testdata_true_vs_predicted_plot(y_test, test_predict,savename):
    sample_count = np.arange(1, len(y_test) + 1)
    plt.figure(figsize=(10, 8))
    plt.plot(sample_count, y_test, label="True Values", marker='o', color=(22/255, 6/255, 138/255), linewidth=1.5)
    plt.plot(sample_count, test_predict, label="Predicted Values", marker='x', color=(239/255, 145/255, 99/255), linewidth=1.5)
    plt.title("True vs Predicted Values for thermal conductivity /W，m-1，K-1")
    plt.xlabel("Sample Count")
    plt.ylabel("Thermal conductivity (W，m-1，K-1) ")
    plt.legend()
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format='png', dpi=600, bbox_inches='tight')
    png_buffer.seek(0)
    img = Image.open(png_buffer)
    img.save(savepath+savename, format='TIFF')