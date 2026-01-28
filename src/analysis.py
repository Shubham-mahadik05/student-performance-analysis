import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/student_performance.csv")

# Calculate total and average
df["Total"] = df[["Maths", "Science", "English"]].sum(axis=1)
df["Average"] = df["Total"] / 3

# Sort by average
top_students = df.sort_values(by="Average", ascending=False)

print("\n📊 Student Performance Summary:\n")
print(top_students[["RollNo", "Name", "Average"]])

# Save result
top_students.to_csv("outputs/performance_report.csv", index=False)

# Set style for better-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# ============ Chart 1: Average Performance by Student ============
fig1 = plt.figure(figsize=(12, 6))
plt.bar(df["Name"], df["Average"], color='skyblue', edgecolor='navy')
plt.xlabel("Students", fontsize=12, fontweight='bold')
plt.ylabel("Average Marks", fontsize=12, fontweight='bold')
plt.title("📊 Student Average Performance", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.ylim(0, 100)
for i, v in enumerate(df["Average"]):
    plt.text(i, v + 2, f'{v:.1f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/01_average_performance.png", dpi=300, bbox_inches='tight')
print("✅ Chart 1 saved: Average Performance by Student")

# ============ Chart 2: Subject-wise Performance Comparison ============
fig2 = plt.figure(figsize=(12, 6))
subjects = ["Maths", "Science", "English"]
subject_averages = [df["Maths"].mean(), df["Science"].mean(), df["English"].mean()]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = plt.bar(subjects, subject_averages, color=colors, edgecolor='black', linewidth=2)
plt.ylabel("Average Marks", fontsize=12, fontweight='bold')
plt.title("📚 Subject-wise Average Performance", fontsize=14, fontweight='bold')
plt.ylim(0, 100)
for bar, avg in zip(bars, subject_averages):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{avg:.1f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/02_subject_wise_performance.png", dpi=300, bbox_inches='tight')
print("✅ Chart 2 saved: Subject-wise Performance Comparison")

# ============ Chart 3: Attendance vs Average Performance ============
fig3 = plt.figure(figsize=(12, 6))
plt.scatter(df["Attendance"], df["Average"], s=200, alpha=0.6, c=df["Average"], cmap='viridis', edgecolor='black', linewidth=2)
plt.xlabel("Attendance (%)", fontsize=12, fontweight='bold')
plt.ylabel("Average Marks", fontsize=12, fontweight='bold')
plt.title("🎯 Attendance vs Average Performance", fontsize=14, fontweight='bold')
plt.colorbar(label='Average Marks')
for idx, row in df.iterrows():
    plt.annotate(row['Name'], (row['Attendance'], row['Average']), fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/charts/03_attendance_vs_performance.png", dpi=300, bbox_inches='tight')
print("✅ Chart 3 saved: Attendance vs Average Performance")

# ============ Chart 4: Result Distribution (Pass/Fail) ============
fig4 = plt.figure(figsize=(10, 6))
result_counts = df["Result"].value_counts()
colors_pie = ['#2ECC71', '#E74C3C']
explode = (0.05,) * len(result_counts)
plt.pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%',
        colors=colors_pie, explode=explode, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title("Result Distribution (Pass/Fail)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/04_result_distribution.png", dpi=300, bbox_inches='tight')
print("✅ Chart 4 saved: Result Distribution")

# ============ Chart 5: Subject Performance by Student (Grouped Bar Chart) ============
fig5 = plt.figure(figsize=(14, 6))
x = range(len(df))
width = 0.25
plt.bar([i - width for i in x], df["Maths"], width, label='Maths', color='#FF6B6B')
plt.bar([i for i in x], df["Science"], width, label='Science', color='#4ECDC4')
plt.bar([i + width for i in x], df["English"], width, label='English', color='#45B7D1')
plt.xlabel("Students", fontsize=12, fontweight='bold')
plt.ylabel("Marks", fontsize=12, fontweight='bold')
plt.title("📖 Subject-wise Performance by Student", fontsize=14, fontweight='bold')
plt.xticks(x, df["Name"], rotation=45)
plt.legend(fontsize=11)
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig("outputs/charts/05_subject_by_student.png", dpi=300, bbox_inches='tight')
print("✅ Chart 5 saved: Subject Performance by Student")

# ============ Chart 6: Attendance Distribution ============
fig6 = plt.figure(figsize=(12, 6))
plt.bar(df["Name"], df["Attendance"], color='#9B59B6', edgecolor='black', linewidth=2)
plt.xlabel("Students", fontsize=12, fontweight='bold')
plt.ylabel("Attendance (%)", fontsize=12, fontweight='bold')
plt.title("📅 Student Attendance Distribution", fontsize=14, fontweight='bold')
plt.axhline(y=75, color='r', linestyle='--', linewidth=2, label='Min. Attendance (75%)')
plt.xticks(rotation=45)
plt.ylim(0, 105)
plt.legend()
for i, v in enumerate(df["Attendance"]):
    plt.text(i, v + 2, f'{v}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/06_attendance_distribution.png", dpi=300, bbox_inches='tight')
print("✅ Chart 6 saved: Attendance Distribution")

# ============ Chart 7: Correlation Heatmap ============
fig7 = plt.figure(figsize=(10, 8))
numeric_cols = ["Maths", "Science", "English", "Attendance", "Average"]
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, square=True, 
            linewidths=2, cbar_kws={"shrink": 0.8}, fmt='.2f', annot_kws={'size': 10})
plt.title("🔗 Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/charts/07_correlation_heatmap.png", dpi=300, bbox_inches='tight')
print("✅ Chart 7 saved: Correlation Heatmap")

# ============ Chart 8: Box Plot for Subject Scores ============
fig8 = plt.figure(figsize=(12, 6))
data_to_plot = [df["Maths"], df["Science"], df["English"]]
bp = plt.boxplot(data_to_plot, labels=["Maths", "Science", "English"], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4', '#45B7D1']):
    patch.set_facecolor(color)
plt.ylabel("Marks", fontsize=12, fontweight='bold')
plt.title("📦 Subject Score Distribution (Box Plot)", fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/charts/08_subject_boxplot.png", dpi=300, bbox_inches='tight')
print("✅ Chart 8 saved: Subject Box Plot")

# ============ Chart 9: Total Marks Distribution ============
fig9 = plt.figure(figsize=(12, 6))
plt.barh(df["Name"], df["Total"], color='#F39C12', edgecolor='black', linewidth=2)
plt.xlabel("Total Marks", fontsize=12, fontweight='bold')
plt.title("🏆 Student Total Marks Distribution", fontsize=14, fontweight='bold')
for i, v in enumerate(df["Total"]):
    plt.text(v + 5, i, f'{v}', va='center', fontweight='bold')
plt.xlim(0, 310)
plt.tight_layout()
plt.savefig("outputs/charts/09_total_marks.png", dpi=300, bbox_inches='tight')
print("✅ Chart 9 saved: Total Marks Distribution")

# ============ Chart 10: Performance Summary Statistics ============
fig10, axes = plt.subplots(2, 2, figsize=(14, 10))

# Average by Result
ax1 = axes[0, 0]
result_avg = df.groupby("Result")["Average"].mean()
result_avg.plot(kind='bar', ax=ax1, color=['#2ECC71', '#E74C3C'], edgecolor='black')
ax1.set_title("Average Performance by Result", fontweight='bold')
ax1.set_ylabel("Average Marks")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
for i, v in enumerate(result_avg):
    ax1.text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

# Attendance comparison
ax2 = axes[0, 1]
df.sort_values('Attendance', ascending=False).plot(x='Name', y='Attendance', ax=ax2, kind='barh', color='#9B59B6', edgecolor='black', legend=False)
ax2.set_title("Attendance Comparison", fontweight='bold')
ax2.set_xlabel("Attendance (%)")

# Statistics text
ax3 = axes[1, 0]
ax3.axis('off')
stats_text = f"""
📊 PERFORMANCE STATISTICS

Total Students: {len(df)}
Pass: {(df['Result'] == 'Pass').sum()}
Fail: {(df['Result'] == 'Fail').sum()}

Average Marks: {df['Average'].mean():.2f}
Highest Score: {df['Average'].max():.2f}
Lowest Score: {df['Average'].min():.2f}

Avg Attendance: {df['Attendance'].mean():.2f}%
"""
ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), verticalalignment='center')

# Subject average comparison
ax4 = axes[1, 1]
subject_data = {'Maths': df['Maths'].mean(), 'Science': df['Science'].mean(), 'English': df['English'].mean()}
ax4.bar(subject_data.keys(), subject_data.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black')
ax4.set_title("Subject Average Comparison", fontweight='bold')
ax4.set_ylabel("Average Marks")
ax4.set_ylim(0, 100)
for i, (k, v) in enumerate(subject_data.items()):
    ax4.text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/charts/10_performance_summary.png", dpi=300, bbox_inches='tight')
print("✅ Chart 10 saved: Performance Summary")

print("\n" + "="*50)
print("📈 Analysis Complete!")
print("="*50)
print("All charts have been saved to outputs/charts/")
print("="*50)
