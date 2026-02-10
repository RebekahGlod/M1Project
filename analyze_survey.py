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

    # Store the question text mapping
    question_map = {col[0]: col[1] for col in df.columns}

    # Remove metadata row
    if isinstance(df.iloc[0, 0], str) and df.iloc[0, 0].startswith('{"ImportId"'):
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = df.columns.droplevel(1)
    return df, question_map

def area_1_enrollment_threat(df):
    """
    Analysis Area 1: The Enrollment "Threat"
    """
    print("--- Analysis Area 1: Enrollment Threat ---")

    # Filter for Graduate Students
    grad_students = df[df['Q27'] == 'Graduate'].copy()
    total_grad = len(grad_students)
    print(f"Total Graduate Students: {total_grad}")

    # Calculate % Unaware of alternative pathway before program (Q31)
    unaware_grad = grad_students[grad_students['Q31'] == 'No'].copy()
    num_unaware = len(unaware_grad)
    pct_unaware = (num_unaware / total_grad) * 100 if total_grad > 0 else 0
    print(f"Unaware of Alternative Pathway: {num_unaware} ({pct_unaware:.1f}%)")

    # Analyze Q35 for unaware group: Likelihood to enroll if known earlier
    # Map likert to numeric for sorting/plotting if needed, or just count categories
    # Q35: 'Extremely likely', 'Somewhat likely', 'Neither likely nor unlikely', 'Somewhat unlikely', 'Extremely unlikely'

    q35_map = {
        'Extremely unlikely': 1,
        'Somewhat unlikely': 2,
        'Neither likely nor unlikely': 3,
        'Somewhat likely': 4,
        'Extremely likely': 5
    }

    # Normalize categories just in case of whitespace
    unaware_grad['Q35_numeric'] = unaware_grad['Q35'].map(q35_map)

    # "Potential Enrollment Loss": Unlikely (1) or Somewhat Unlikely (2)
    potential_loss_count = unaware_grad[unaware_grad['Q35_numeric'] <= 2].shape[0]
    pct_potential_loss_of_unaware = (potential_loss_count / num_unaware) * 100 if num_unaware > 0 else 0

    # Calculate percentage of ALL graduate students who are at risk (Unaware AND Unlikely)
    pct_potential_loss_total = (potential_loss_count / total_grad) * 100 if total_grad > 0 else 0

    print(f"Potential Enrollment Loss (Unaware & Unlikely to Enroll): {potential_loss_count}")
    print(f"% of Unaware Group: {pct_potential_loss_of_unaware:.1f}%")
    print(f"% of Total Graduate Students: {pct_potential_loss_total:.1f}%")

    # Visualization
    # Bar chart of Q35 responses for the unaware group
    counts = unaware_grad['Q35'].value_counts().reindex([
        'Extremely unlikely', 'Somewhat unlikely', 'Neither likely nor unlikely',
        'Somewhat likely', 'Extremely likely'
    ], fill_value=0)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title(f"Likelihood to Enroll if Alternative Pathway Known\n(Among {num_unaware} Unaware Graduate Students)")
    plt.ylabel("Number of Students")
    plt.xlabel("Likelihood")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('enrollment_threat.png')
    plt.close()

    return {
        'Total_Grad_Students': total_grad,
        'Pct_Unaware': pct_unaware,
        'Pct_Potential_Loss_of_Unaware': pct_potential_loss_of_unaware,
        'Pct_Potential_Loss_Total': pct_potential_loss_total
    }

def area_2_roi_pressure(df):
    """
    Analysis Area 2: ROI vs. Employer Pressure
    """
    print("\n--- Analysis Area 2: ROI vs. Employer Pressure ---")

    # Combine Q55 and Q44 for ROI Belief
    df['ROI_Belief_Raw'] = df['Q55'].fillna(df['Q44'])

    # Map to numeric
    roi_map = {
        'Definitely not': 1,
        'Probably not': 2,
        'Might or might not': 3,
        'Probably yes': 4,
        'Definitely yes': 5
    }
    df['ROI_Belief_Score'] = df['ROI_Belief_Raw'].map(roi_map)

    # Employer Pressure (Q49) - Clean Yes/No
    # Filter out NaNs in Q49 for this analysis
    roi_df = df.dropna(subset=['ROI_Belief_Score', 'Q49']).copy()

    # Segment "Reluctant Students": Low ROI (<= 3) AND Employer Pressure (Yes)
    reluctant = roi_df[(roi_df['ROI_Belief_Score'] <= 3) & (roi_df['Q49'] == 'Yes')]
    num_reluctant = len(reluctant)
    pct_reluctant = (num_reluctant / len(roi_df)) * 100

    print(f"Total students in ROI analysis: {len(roi_df)}")
    print(f"Reluctant Students (Low ROI Belief & High Pressure): {num_reluctant} ({pct_reluctant:.1f}%)")

    # Visualization: ROI Belief by Employer Pressure
    plt.figure(figsize=(10, 6))

    # Calculate proportions for better comparison
    props = roi_df.groupby('Q49')['ROI_Belief_Raw'].value_counts(normalize=True).unstack(fill_value=0)
    props = props.reindex(['Definitely not', 'Probably not', 'Might or might not', 'Probably yes', 'Definitely yes'], axis=1).fillna(0)

    props.plot(kind='bar', stacked=True, colormap='RdYlGn', figsize=(10, 6))
    plt.title("ROI Belief by Employer Requirement")
    plt.ylabel("Proportion of Students")
    plt.xlabel("Employer Requirement (Q49)")
    plt.legend(title="Belief in Higher Earnings", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('roi_vs_pressure.png')
    plt.close()

    # Heatmap alternative (Confusion Matrix style)
    # Pivot table of counts
    heatmap_data = roi_df.groupby(['Q49', 'ROI_Belief_Raw']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(['Definitely not', 'Probably not', 'Might or might not', 'Probably yes', 'Definitely yes'], axis=1)

    return {
        'Num_Reluctant_Students': num_reluctant,
        'Pct_Reluctant_Students': pct_reluctant
    }

def area_3_program_value(df):
    """
    Analysis Area 3: Irreplaceability (Program Value)
    """
    print("\n--- Analysis Area 3: Program Value ---")

    cols = ['Q24_1', 'Q24_2', 'Q24_3', 'Q24_4', 'Q24_5', 'Q24_6']
    labels = {
        'Q24_1': 'CPA Prep',
        'Q24_2': 'Networking',
        'Q24_3': 'Faculty Interaction',
        'Q24_4': 'Technical Skills',
        'Q24_5': 'Soft Skills',
        'Q24_6': 'Recruiting'
    }

    # Convert to numeric
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate mean rank (Lower is better/harder to replace)
    mean_ranks = df[cols].mean().sort_values()

    print("Mean Ranks (Lower = Harder to Replace):")
    print(mean_ranks)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mean_ranks.values, y=[labels[c] for c in mean_ranks.index], palette="rocket")
    plt.title("Hardest to Replace Program Features (Lower Rank = More Essential)")
    plt.xlabel("Mean Rank (1=Most Important, 6=Least)")
    plt.tight_layout()
    plt.savefig('program_value.png')
    plt.close()

    return {
        'Top_Feature': labels[mean_ranks.index[0]],
        'Top_Feature_Score': mean_ranks.iloc[0],
        'Bottom_Feature': labels[mean_ranks.index[-1]],
        'Bottom_Feature_Score': mean_ranks.iloc[-1]
    }

def export_summary(metrics_1, metrics_2, metrics_3):
    """
    Exports summary to CSV.
    """
    summary_data = {
        'Metric': [
            'Total Graduate Students',
            'Pct Unaware of Alt Pathway',
            'Pct Unaware who would be Unlikely to Enroll',
            'Pct Total Grad Students at Risk',
            'Number of Reluctant Students (Low ROI, High Pressure)',
            'Pct Reluctant Students',
            'Most Essential Feature (Hardest to Replace)',
            'Least Essential Feature'
        ],
        'Value': [
            metrics_1['Total_Grad_Students'],
            f"{metrics_1['Pct_Unaware']:.1f}%",
            f"{metrics_1['Pct_Potential_Loss_of_Unaware']:.1f}%",
            f"{metrics_1['Pct_Potential_Loss_Total']:.1f}%",
            metrics_2['Num_Reluctant_Students'],
            f"{metrics_2['Pct_Reluctant_Students']:.1f}%",
            f"{metrics_3['Top_Feature']} (Rank: {metrics_3['Top_Feature_Score']:.2f})",
            f"{metrics_3['Bottom_Feature']} (Rank: {metrics_3['Bottom_Feature_Score']:.2f})"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('cpa_pathway_summary.csv', index=False)
    print("\nSummary exported to 'cpa_pathway_summary.csv'")

if __name__ == "__main__":
    filepath = 'Alternative CPA Pathways Survey_December 31, 2025_09.45.csv'
    df, _ = load_and_clean_data(filepath)

    m1 = area_1_enrollment_threat(df)
    m2 = area_2_roi_pressure(df)
    m3 = area_3_program_value(df)

    export_summary(m1, m2, m3)
