# Create a figure with four subplots
plt.figure(figsize=(12, 6))

# First subplot
plt.subplot(2, 2, 1)  # 2 rows, 2 columns, 1st subplot
plt.scatter(lambda_values, results_lambda, marker='o')
plt.title('Optimal Object Function Values for Different λ Values', fontsize=12)
# plt.xlabel('Lambda')
plt.ylabel('Optimal Value', fontsize=12)
plt.xticks(lambda_values)
plt.grid()

# Second subplot
plt.subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd subplot
for i in range(resultsw_lambda.shape[0]):
    plt.scatter(np.arange(1, 6), resultsw_lambda[i, :-1], label=f'λ = {lambda_values[i]:.1f}')
plt.title('Optimal Portfolio Allocation w for Different λ Values', fontsize=12)
# plt.xlabel('w')
plt.ylabel('Optimal Solution Values', fontsize=12)
plt.legend()
plt.grid()

# Third subplot
plt.subplot(2, 2, 3)  # 2 rows, 2 columns, 3rd subplot
plt.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing
plt.scatter(beta_values, results_beta, marker='o')
plt.title('Optimal Object Function Values for Different $\\beta$ Values', fontsize=12)
# plt.xlabel('$\\beta$')
plt.ylabel('Optimal Value', fontsize=12)
plt.xticks(beta_values)
plt.grid()

# Fourth subplot
plt.subplot(2, 2, 4)  # 2 rows, 2 columns, 4th subplot
plt.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing
for i in range(resultsw_beta.shape[0]):
    plt.scatter(np.arange(1, 6), resultsw_beta[i, :-1], label=f'$\\beta$ = {beta_values[i]:.1f}')
plt.title('Optimal Portfolio Allocation w for Different $\\beta$ Values', fontsize=12)
# plt.xlabel('w')
plt.ylabel('Optimal Solution Values', fontsize=12)
plt.legend()
plt.grid()

# Overall title for the figure
plt.suptitle('Box Uncertainty Set', fontsize=16, y=0.98)

# Show the plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.savefig('box_uncertainty_set.pdf', dpi=800)