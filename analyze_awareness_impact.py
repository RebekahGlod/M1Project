import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def load_and_clean_data(filepath):
    """
    Loads the survey data, handles headers, and filters metadata.
    """
    df = pd.read_csv(filepath, header=[0, 1])

    # Remove metadata row if present
    if isinstance(df.iloc[0, 0], str) and df.iloc[0, 0].startswith('{"ImportId"'):
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = df.columns.droplevel(1)

    # Filter out rows where key questions are blank if necessary, but keep as much data as possible
    # We will filter per analysis section to maximize data usage
    return df

def analyze_awareness_gap(df):
    """
    Step 2: The 'Awareness Gap' Analysis
    """
    print("--- Analysis Step 2: Awareness Gap ---")

    # 1. Grad Students (Q27='Graduate') - Q31 Awareness
    # Filter out blank responses for Q31
    grad_df = df[(df['Q27'] == 'Graduate') & (df['Q31'].notna())].copy()
    total_grad = len(grad_df)
    grad_aware_counts = grad_df['Q31'].value_counts()

    unaware_grad_count = grad_aware_counts.get('No', 0)
    aware_grad_count = grad_aware_counts.get('Yes', 0)

    pct_unaware_grad = (unaware_grad_count / total_grad) * 100 if total_grad > 0 else 0
    pct_aware_grad = (aware_grad_count / total_grad) * 100 if total_grad > 0 else 0

    print(f"Graduate Students (Total valid responses: {total_grad})")
    print(f"  Unaware: {unaware_grad_count} ({pct_unaware_grad:.1f}%)")
    print(f"  Aware: {aware_grad_count} ({pct_aware_grad:.1f}%)")

    # 2. Undergrad Students (Q27='Undergraduate') - Q53 Awareness
    # Filter out blank responses for Q53
    undergrad_df = df[(df['Q27'] == 'Undergraduate') & (df['Q53'].notna())].copy()
    total_undergrad = len(undergrad_df)
    undergrad_aware_counts = undergrad_df['Q53'].value_counts()

    unaware_undergrad_count = undergrad_aware_counts.get('No', 0)
    aware_undergrad_count = undergrad_aware_counts.get('Yes', 0)

    pct_unaware_undergrad = (unaware_undergrad_count / total_undergrad) * 100 if total_undergrad > 0 else 0
    pct_aware_undergrad = (aware_undergrad_count / total_undergrad) * 100 if total_undergrad > 0 else 0

    print(f"Undergraduate Students (Total valid responses: {total_undergrad})")
    print(f"  Unaware: {unaware_undergrad_count} ({pct_unaware_undergrad:.1f}%)")
    print(f"  Aware: {aware_undergrad_count} ({pct_aware_undergrad:.1f}%)")

    # 3. Visualization: Side-by-side bar chart
    data = {
        'Group': ['Graduate', 'Graduate', 'Undergraduate', 'Undergraduate'],
        'Status': ['Aware', 'Unaware', 'Aware', 'Unaware'],
        'Count': [aware_grad_count, unaware_grad_count, aware_undergrad_count, unaware_undergrad_count],
        'Percentage': [pct_aware_grad, pct_unaware_grad, pct_aware_undergrad, pct_unaware_undergrad]
    }
    viz_df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Group', y='Percentage', hue='Status', data=viz_df, palette=['#2ecc71', '#e74c3c'])
    plt.title('Awareness of Alternative CPA Pathway')
    plt.ylabel('Percentage of Students')
    plt.ylim(0, 100)

    # Add text labels on bars
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:  # Only label if visible
            plt.gca().text(p.get_x() + p.get_width()/2., height + 1,
                           f'{height:.1f}%', ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig('awareness_gap.png')
    plt.close()
    print("Saved 'awareness_gap.png'")

    return {
        'grad_unaware_pct': pct_unaware_grad,
        'undergrad_unaware_pct': pct_unaware_undergrad
    }

def analyze_incentive_shift(df):
    """
    Step 3: The 'Incentive Shift' Analysis
    """
    print("\n--- Analysis Step 3: Incentive Shift ---")

    # 1. Desire Impact (Q52) - Undergraduates only
    # Filter for Undergrads who answered Q52
    undergrad_q52 = df[(df['Q27'] == 'Undergraduate') & (df['Q52'].notna())].copy()

    print(f"Undergraduates responding to Q52: {len(undergrad_q52)}")

    q52_counts = undergrad_q52['Q52'].value_counts()
    print("Q52 Distribution (Impact on Desire to Pursue Grad Degree):")
    print(q52_counts)

    # Visualization: Pie Chart for Q52
    plt.figure(figsize=(8, 8))
    # Filter out small slices or group them if needed, but for now show all
    # Use a distinct color palette
    colors = sns.color_palette('pastel')[0:len(q52_counts)]
    plt.pie(q52_counts, labels=q52_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Impact of Alternative Pathway Knowledge on Desire to Pursue Graduate Degree\n(Undergraduates)')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    plt.savefig('incentive_shift_pie.png')
    plt.close()
    print("Saved 'incentive_shift_pie.png'")

    # 2. The "Regret" Factor
    # Note: Using Q35 ("Likelihood to enroll if known") instead of Q33 ("Career ladder agreement")
    # based on header inspection matching the intent of "Likelihood of pursuing a graduate degree".

    # Filter Grads who were Unaware (Q31='No') AND answered Q35
    unaware_grads = df[(df['Q27'] == 'Graduate') & (df['Q31'] == 'No') & (df['Q35'].notna())].copy()
    total_unaware_grads = len(unaware_grads)

    # Q35: Likelihood to enroll if known earlier
    # Categories: 'Extremely unlikely', 'Somewhat unlikely', 'Neither likely nor unlikely', 'Somewhat likely', 'Extremely likely'
    q35_counts = unaware_grads['Q35'].value_counts()

    unlikely_count = q35_counts.get('Extremely unlikely', 0) + q35_counts.get('Somewhat unlikely', 0)
    pct_unlikely = (unlikely_count / total_unaware_grads) * 100 if total_unaware_grads > 0 else 0

    print(f"Unaware Graduate Students (Valid Q35 Responses): {total_unaware_grads}")
    print(f"Would have been Unlikely to Enroll if known (Regret Factor): {unlikely_count} ({pct_unlikely:.1f}%)")

    # 3. Correlation: Q52 vs Awareness (Q53) for Undergrads
    # Compare Q52 distribution between Aware and Unaware Undergrads
    # Ensure Q53 is not NaN for this comparison
    undergrad_aware = df[(df['Q27'] == 'Undergraduate') & (df['Q53'] == 'Yes') & (df['Q52'].notna())]
    undergrad_unaware = df[(df['Q27'] == 'Undergraduate') & (df['Q53'] == 'No') & (df['Q52'].notna())]

    print("\n--- Correlation: Q52 (Incentive Shift) by Awareness (Q53) ---")
    print(f"Aware Undergrads (Q53=Yes): {len(undergrad_aware)}")
    print(undergrad_aware['Q52'].value_counts(normalize=True).mul(100).round(1))

    print(f"\nUnaware Undergrads (Q53=No): {len(undergrad_unaware)}")
    print(undergrad_unaware['Q52'].value_counts(normalize=True).mul(100).round(1))

    return {
        'regret_pct': pct_unlikely
    }

def main():
    filepath = 'Alternative CPA Pathways Survey_December 31, 2025_09.45.csv'
    df = load_and_clean_data(filepath)

    metrics_gap = analyze_awareness_gap(df)
    metrics_incentive = analyze_incentive_shift(df)

    # Write summary to text file
    with open('awareness_impact_summary.txt', 'w') as f:
        f.write("--- Analysis Summary ---\n")
        f.write(f"Graduate Unaware %: {metrics_gap['grad_unaware_pct']:.1f}%\n")
        f.write(f"Undergraduate Unaware %: {metrics_gap['undergrad_unaware_pct']:.1f}%\n")
        f.write(f"Regret Factor (Unaware Grads who would be Unlikely to Enroll): {metrics_incentive['regret_pct']:.1f}%\n")

if __name__ == "__main__":
    main()
