import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的ExtraTrees模型
model = joblib.load('cat_grid_search.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "SEX": {"type": "categorical", "options": [0, 1], "default": 0},
    "AGE": {"type": "numerical", "min": 0.0, "max": 18.0, "default": 5.0},
    "WT": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 25.0},
    "Single_Dose": {"type": "numerical", "min": 0.0, "max": 60.0, "default": 15.0},
    "Daily_Dose": {"type": "numerical", "min": 0.0, "max": 2400.0, "default": 450.0},
    "SCR": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 30.0},
    "CLCR": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 90.0},
    "BUN": {"type": "numerical", "min": 0.0, "max": 50.0, "default": 5.0},
    "ALT": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 18.0},
    "AST": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 18.0},
    "CL": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 3.85},
    "V": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 10.0}
}
# Streamlit 界面
st.title("Antiepileptic Drug (OXC) Treatment Outcome Prediction with SHAP Visualization")

# Description 
st.write("""
This app predicts the likelihood of antiepileptic drug (OXC) treatment outcome based on input features.
Select the Catboost model, input feature values, and get predictions and probability estimates.
""")


# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    prediction = model.predict(features)[0]  # 预测结果是连续性变量

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    st.header("Prediction Result")
    st.success(f"Based on the feature values, the predicted concentration is {prediction:.2f} mg/L.")

    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

