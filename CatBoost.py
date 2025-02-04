import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 加载模型
model_path = "cat_grid_search.pkl"
model = joblib.load(model_path)

# 获取最佳估计器
best_model = model.best_estimator_

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="Concentration Prediction", page_icon="📊")
st.title("📊 Concentration Prediction and SHAP Visualization")
st.write("""
通过输入特征值，您可以获取模型的预测结果，并通过 SHAP 分析了解每个特征的贡献。
""")

# 特征输入区域
st.sidebar.header("Feature Input Area")
st.sidebar.write("请输入特征值：")

# 定义特征输入范围
feature_ranges = {
    "SEX": {"type": "categorical", "options": [0, 1], "default": 0, "description": "性别 (0 = 女, 1 = 男)"},
    "AGE": {"type": "numerical", "min": 0.0, "max": 18.0, "default": 5.0, "description": "患者年龄 (岁)"},
    "WT": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 25.0, "description": "患者体重 (kg)"},
    "Single_Dose": {"type": "numerical", "min": 0.0, "max": 60.0, "default": 15.0, "description": "单次给药剂量/体重 (mg/kg)"},
    "Daily_Dose": {"type": "numerical", "min": 0.0, "max": 2400.0, "default": 450.0, "description": "日总剂量 (mg)"},
    "SCR": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 30.0, "description": "血清肌酐水平 (μmol/L)"},
    "CLCR": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 90.0, "description": "肌酐清除率 (L/h)"},
    "BUN": {"type": "numerical", "min": 0.0, "max": 50.0, "default": 5.0, "description": "血尿素氮水平 (mmol/L)"},
    "ALT": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 18.0, "description": "丙氨酸氨基转移酶水平 (U/L)"},
    "AST": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 18.0, "description": "天冬氨酸氨基转移酶水平 (U/L)"},
    "CL": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 3.85, "description": "药物的代谢清除率 (L/h)"},
    "V": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 10.0, "description": "药物的表观分布容积 (L)"}
}

# 动态生成输入界面
inputs = {}
for feature, config in feature_ranges.items():
    if config["type"] == "numerical":
        inputs[feature] = st.sidebar.number_input(
            f"{feature} ({config['description']})",
            min_value=config["min"],
            max_value=config["max"],
            value=config["default"]
        )
    elif config["type"] == "categorical":
        inputs[feature] = st.sidebar.selectbox(
            f"{feature} ({config['description']})",
            options=config["options"],
            index=config["options"].index(config["default"])
        )

# 将输入特征转换为 Pandas DataFrame
features_df = pd.DataFrame([inputs])

# 如果模型在训练时使用了分类特征，确保这些特征是整数类型
cat_features = ["SEX"]  # 假设 SEX 是分类特征
features_df[cat_features] = features_df[cat_features].astype(int)

# 模型预测
if st.button("Predict"):
    try:
        prediction = best_model.predict(features_df)[0]  # 预测结果是连续性变量

        # 显示预测结果
        st.header("Prediction Result")
        st.success(f"Based on the feature values, the predicted concentration is {prediction:.2f} mg/L.")

        # 保存预测结果为图像
        fig, ax = plt.subplots(figsize=(8, 1))
        text = f"Predicted Concentration: {prediction:.2f} mg/L"
        ax.text(
            0.5, 0.5, text,
            fontsize=16,
            ha='center', va='center',
            fontname='Times New Roman',
            transform=ax.transAxes
        )
        ax.axis('off')
        plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
        st.image("prediction_text.png", use_column_width=True)

        # 计算 SHAP 值
        try:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(features_df)

            # 生成 SHAP 力图
            st.header("SHAP Force Plot")
            html_output = shap.force_plot(
                explainer.expected_value,
                shap_values[0, :],
                features_df.iloc[0, :],
                show=False
            )
            shap_html = f"<head>{shap.getjs()}</head><body>{html_output.html()}</body>"
            st.components.v1.html(shap_html, height=400)

            # 生成 SHAP 摘要图
            st.header("SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, features_df, plot_type="dot", show=False)
            plt.title("SHAP Values for Each Feature")
            st.pyplot(fig)

            # 生成 SHAP 特征重要性排序图
            st.header("SHAP Feature Importance")
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, features_df, plot_type="bar", show=False)
            plt.title("SHAP Values for Each Feature")
            st.pyplot(fig)

            # 生成 SHAP 决策图
            st.header("SHAP Decision Plot")
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.decision_plot(explainer.expected_value, shap_values[0, :], features_df.iloc[0, :], show=False)
            plt.title("SHAP Decision Plot")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred during SHAP visualization: {e}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# 预测准确性图
st.header("Prediction Accuracy")
st.write("展示模型的绝对准确度和相对准确度。")

# 假设你有真实值和预测值
true_values = [1.0, 2.0, 3.0, 4.0, 5.0]  # 真实值
predicted_values = [1.1, 2.1, 2.9, 4.1, 5.1]  # 预测值

# 绘制散点图
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(true_values, predicted_values, alpha=0.5, color='blue', label='Predictions')
ax.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linestyle='--', label='Ideal Line')
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Prediction Accuracy')
ax.legend()

# 添加指标信息
mae = mean_absolute_error(true_values, predicted_values)
mse = mean_squared_error(true_values, predicted_values)
r2 = r2_score(true_values, predicted_values)

textstr = '\n'.join((
    f'MAE: {mae:.2f}',
    f'MSE: {mse:.2f}',
    f'R²: {r2:.2f}'))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.9
