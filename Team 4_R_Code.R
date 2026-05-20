# =====================================================================
# HR EMPLOYEE ATTRITION PREDICTION PROJECT
# =====================================================================
# Team 4 Project
# Team Members:  
#   - Rithik Rathinavel Ragupathi 
#   - Irfan Saleemudeen 
#   - Sashank Addanki Venkata Naga 
#   - Lakshmi Srujana Sushma Pedapati
# =====================================================================

# ==============================================
# PART 1: LOAD REQUIRED LIBRARIES
# ==============================================
library(tidyverse)
library(caret)
library(randomForest)
library(ROCR)
library(corrplot)
library(gridExtra)
library(ggplot2)
library(scales)
library(rpart)
library(rpart.plot)

# ==============================================
# PART 2: LOAD DATA
# ==============================================
hr_data <- read.csv("HR-Employee-Attrition_Data set.csv")
cat("Dimensions:", nrow(hr_data), "rows,", ncol(hr_data), "columns\n")

# ==============================================
# PART 3 : Data Cleaning and Pre-processing
# ==============================================
# Remove unnecessary columns
hr_data <- hr_data %>%
  select(-EmployeeNumber,
         -EmployeeCount,
         -Over18,
         -StandardHours)

# Convert target variable to factor
hr_data$Attrition <- factor(hr_data$Attrition, levels=c("No", "Yes"))

# Convert character variables to factors
hr_data <- hr_data %>%
  mutate_if(is.character, as.factor)

# Check cleaned structure
str(hr_data)

# ==============================================
# PART 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================
print(head(hr_data))
str(hr_data)
summary(hr_data)
print(sort(colSums(is.na(hr_data)), decreasing = TRUE))
cat("Duplicate rows:", sum(duplicated(hr_data)), "\n")

target_dist <- hr_data %>%
  count(Attrition) %>%
  mutate(Percent = percent(n / sum(n)))
print(target_dist)

# ==============================================
# PART 5: DATA VISUALIZATION (EDA Plots)
# ==============================================
# Plot 1: Attrition Distribution
attrition_counts <- table(hr_data$Attrition)
colors <- c("red", "green")
pie(attrition_counts, 
    labels = c(paste("No\n(", attrition_counts["No"], ")", sep=""),
               paste("Yes\n(", attrition_counts["Yes"], ")", sep="")),
    main = "Employee Attrition Distribution",
    col = colors,
    cex = 1.2,
    font = 2)

# Plot 2: Attrition by Department
dept_attrition <- prop.table(table(hr_data$Department, hr_data$Attrition), margin = 1) * 100
barplot(dept_attrition[, "Yes"], 
        main = "Attrition Rate by Department",
        xlab = "Department",
        ylab = "Attrition Rate (%)",
        col = c("blue", "red", "yellow"),
        ylim = c(0, 50))

# Plot 3: Age vs Attrition
boxplot(hr_data$Age ~ hr_data$Attrition,
        main = "Age vs Attrition",
        xlab = "Attrition",
        ylab = "Age",
        col = c("green", "red"))

# Plot 4: Income vs Attrition
boxplot(hr_data$MonthlyIncome ~ hr_data$Attrition,
        main = "Monthly Income vs Attrition",
        xlab = "Attrition",
        ylab = "Monthly Income ($)",
        col = c("green", "red"))

# Plot 5: Years at Company vs Attrition
boxplot(hr_data$YearsAtCompany ~ hr_data$Attrition,
        main = "Years at Company vs Attrition",
        xlab = "Attrition",
        ylab = "Years at Company",
        col = c("green", "red"))

# Plot 6: Job Satisfaction vs Attrition
satisfaction_attrition <- prop.table(table(hr_data$JobSatisfaction, hr_data$Attrition), margin = 1) * 100
barplot(satisfaction_attrition[, "Yes"],
        main = "Attrition Rate by Job Satisfaction Level",
        xlab = "Job Satisfaction (1=Low, 4=High)",
        ylab = "Attrition Rate (%)",
        col = c("red", "orange", "blue", "green"),
        ylim = c(0, max(satisfaction_attrition[, "Yes"]) + 5))

# Plot 7: Overtime vs Attrition
overtime_attrition <- prop.table(table(hr_data$OverTime, hr_data$Attrition), margin = 1) * 100
barplot(overtime_attrition[, "Yes"],
        main = "Attrition Rate by Overtime Status",
        xlab = "Overtime",
        ylab = "Attrition Rate (%)",
        col = c("green", "red"),
        ylim = c(0, max(overtime_attrition[, "Yes"]) + 5))

# Plot 8: Correlation Heatmap
num_data <- hr_data[, sapply(hr_data, is.numeric)]
correlation_matrix <- cor(num_data, use = "pairwise.complete.obs")
correlation_matrix[!is.finite(correlation_matrix)] <- 0

corrplot(correlation_matrix,
         method = "color",
         type = "lower",
         order = "hclust",
         tl.cex = 0.7,
         tl.col = "black",
         title = "Correlation Matrix - Numerical Features",
         mar = c(0,0,2,0))

# ==============================================
# PART 6: DATA PREPROCESSING LEVEL 2
# ==============================================
Attrition_target <- hr_data$Attrition

hr_data_clean <- hr_data %>% 
  select(-Attrition)

X <- model.matrix(~., data = hr_data_clean)[, -1]

colnames(X) <- make.names(colnames(X), unique = TRUE)

data_numeric <- as.data.frame(X)

data_numeric$Attrition <- as.factor(ifelse(Attrition_target == "Yes", 1, 0))

cat("Final dataset shape:", nrow(data_numeric), "rows,", ncol(data_numeric), "columns\n")
print(table(data_numeric$Attrition))

# ==============================================
# PART 7: TRAIN-TEST SPLIT
# ==============================================
set.seed(123)
train_index <- createDataPartition(data_numeric$Attrition, p = 0.7, list = FALSE)
train_data <- data_numeric[train_index, ]
test_data <- data_numeric[-train_index, ]

cat("Training set size:", nrow(train_data), "rows\n")
cat("Test set size:", nrow(test_data), "rows\n")
cat("\nOriginal Training Set - Class Distribution:\n")
print(table(train_data$Attrition))
print(prop.table(table(train_data$Attrition)))

# ==============================================
# PART 8: CLASS IMBALANCE HANDLING (OVERSAMPLING)
# ==============================================
cat("\n=== HANDLING CLASS IMBALANCE ===\n")

# Separate majority and minority classes
majority_class <- train_data[train_data$Attrition == 0, ]
minority_class <- train_data[train_data$Attrition == 1, ]

cat("Original distribution:\n")
cat("Majority (No):", nrow(majority_class), "\n")
cat("Minority (Yes):", nrow(minority_class), "\n")

# Oversample minority class to match majority class (50-50 balance)
set.seed(123)
minority_oversampled <- minority_class[
  sample(nrow(minority_class), 
         nrow(majority_class),
         replace = TRUE),
]

# Combine back
train_data_balanced <- rbind(majority_class, minority_oversampled)

cat("\nBalanced distribution:\n")
cat("Majority (No):", nrow(majority_class), "\n")
cat("Minority (Yes):", nrow(minority_oversampled), "\n")
cat("Total balanced samples:", nrow(train_data_balanced), "\n")

cat("\nBalanced Training Set - Class Distribution:\n")
print(table(train_data_balanced$Attrition))
print(prop.table(table(train_data_balanced$Attrition)))

# ==============================================
# PART 9: MODEL 1 - LOGISTIC REGRESSION
# ==============================================
cat("\n=== MODEL 1: LOGISTIC REGRESSION ===\n")

set.seed(123)
logistic_model <- glm(Attrition ~ ., 
                      data = train_data_balanced,
                      family = "binomial")

logistic_pred_prob <- predict(logistic_model, 
                              newdata = test_data, 
                              type = "response")
logistic_pred_class <- as.factor(ifelse(logistic_pred_prob > 0.5, 1, 0))

logistic_cm <- confusionMatrix(logistic_pred_class, 
                               test_data$Attrition, 
                               positive = "1")
print(logistic_cm)

logistic_accuracy <- logistic_cm$overall["Accuracy"]
logistic_precision <- logistic_cm$byClass["Precision"]
logistic_recall <- logistic_cm$byClass["Recall"]
logistic_f1 <- 2 * (logistic_precision * logistic_recall) / 
  (logistic_precision + logistic_recall)
logistic_auc <- performance(prediction(logistic_pred_prob, test_data$Attrition), 
                            measure = "auc")@y.values[[1]]

cat("\nLogistic Regression Metrics:\n")
cat("Accuracy:", round(logistic_accuracy, 4), "\n")
cat("Precision:", round(logistic_precision, 4), "\n")
cat("Recall:", round(logistic_recall, 4), "\n")
cat("F1-Score:", round(logistic_f1, 4), "\n")
cat("AUC:", round(logistic_auc, 4), "\n")

# ==============================================
# PART 10: MODEL 2 - DECISION TREE
# ==============================================
cat("\n=== MODEL 2: DECISION TREE ===\n")

set.seed(123)
decision_tree_model <- rpart(Attrition ~ ., 
                             data = train_data_balanced,
                             method = "class",
                             control = rpart.control(cp = 0.01, 
                                                     minsplit = 20,
                                                     minbucket = 10))

rpart.plot(decision_tree_model, 
           main = "Decision Tree for Employee Attrition",
           type = 4,
           extra = 101,
           fallen.leaves = TRUE,
           cex = 0.8)

decision_tree_pred_prob <- predict(decision_tree_model, 
                                   newdata = test_data, 
                                   type = "prob")[, 2]
decision_tree_pred_class <- as.factor(ifelse(decision_tree_pred_prob > 0.5, 1, 0))

decision_tree_cm <- confusionMatrix(decision_tree_pred_class, 
                                    test_data$Attrition, 
                                    positive = "1")
print(decision_tree_cm)

decision_tree_accuracy <- decision_tree_cm$overall["Accuracy"]
decision_tree_precision <- decision_tree_cm$byClass["Precision"]
decision_tree_recall <- decision_tree_cm$byClass["Recall"]
decision_tree_f1 <- 2 * (decision_tree_precision * decision_tree_recall) / 
  (decision_tree_precision + decision_tree_recall)
decision_tree_auc <- performance(prediction(decision_tree_pred_prob, test_data$Attrition), 
                                 measure = "auc")@y.values[[1]]

cat("\nDecision Tree Metrics:\n")
cat("Accuracy:", round(decision_tree_accuracy, 4), "\n")
cat("Precision:", round(decision_tree_precision, 4), "\n")
cat("Recall:", round(decision_tree_recall, 4), "\n")
cat("F1-Score:", round(decision_tree_f1, 4), "\n")
cat("AUC:", round(decision_tree_auc, 4), "\n")

# ==============================================
# PART 11: BASELINE RANDOM FOREST
# ==============================================
cat("\n=== BASELINE RANDOM FOREST (WITH BALANCED DATA) ===\n")

set.seed(123)
rf_baseline <- randomForest(
  Attrition ~ .,
  data = train_data_balanced,
  ntree = 500,
  importance = TRUE
)

print(rf_baseline)

rf_base_pred <- predict(rf_baseline, test_data, type = "class")
rf_base_pred_prob <- predict(rf_baseline, test_data, type = "prob")[, 2]

rf_base_cm <- confusionMatrix(rf_base_pred, test_data$Attrition, positive = "1")
print(rf_base_cm)

rf_base_accuracy <- rf_base_cm$overall["Accuracy"]
rf_base_precision <- rf_base_cm$byClass["Precision"]
rf_base_recall <- rf_base_cm$byClass["Recall"]
rf_base_f1 <- 2 * (rf_base_precision * rf_base_recall) / (rf_base_precision + rf_base_recall)
rf_base_auc <- performance(prediction(rf_base_pred_prob, test_data$Attrition), 
                           measure = "auc")@y.values[[1]]

cat("\nBaseline Random Forest Metrics:\n")
cat("Accuracy:", round(rf_base_accuracy, 4), "\n")
cat("Precision:", round(rf_base_precision, 4), "\n")
cat("Recall:", round(rf_base_recall, 4), "\n")
cat("F1-Score:", round(rf_base_f1, 4), "\n")
cat("AUC:", round(rf_base_auc, 4), "\n")

# ==============================================
# PART 12 LOGISTIC REGRESSION THRESHOLD TUNING
# ==============================================

thresholds <- seq(0.1, 0.9, by = 0.05)

logit_threshold_results <- data.frame()

for (t in thresholds) {
  
  pred_class <- factor(ifelse(logistic_pred_prob > t, 1, 0),
                       levels = c(0, 1))
  
  cm <- confusionMatrix(pred_class,
                        test_data$Attrition,
                        positive = "1")
  
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  specificity <- cm$byClass["Specificity"]
  
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  logit_threshold_results <- rbind(
    logit_threshold_results,
    data.frame(
      Threshold = t,
      Accuracy = accuracy,
      Precision = precision,
      Recall = recall,
      Specificity = specificity,
      F1_Score = f1
    )
  )
}

logit_threshold_results

# ==============================================
# SELECT BEST THRESHOLD
# PRIORITIZED HIGH RECALL FOR HR ATTRITION
# ==============================================

# Select thresholds with Recall >= 0.80
candidate_thresholds <- logit_threshold_results %>%
  filter(Recall >= 0.80)

# From those thresholds, choose highest F1-score
best_logit_threshold <- candidate_thresholds[
  which.max(candidate_thresholds$F1_Score), ]

best_logit_threshold

# Final selected threshold
best_threshold <- best_logit_threshold$Threshold

# ==============================================
# FINAL TUNED LOGISTIC REGRESSION MODEL
# ==============================================

logistic_pred_class_tuned <- factor(
  ifelse(logistic_pred_prob > best_threshold, 1, 0),
  levels = c(0, 1)
)

# Confusion matrix
logistic_cm_tuned <- confusionMatrix(
  logistic_pred_class_tuned,
  test_data$Attrition,
  positive = "1"
)

print(logistic_cm_tuned)

# ==============================================
# EXTRACT FINAL METRICS
# ==============================================

logistic_tuned_accuracy <- logistic_cm_tuned$overall["Accuracy"]

logistic_tuned_precision <- logistic_cm_tuned$byClass["Precision"]

logistic_tuned_recall <- logistic_cm_tuned$byClass["Recall"]

logistic_tuned_specificity <- logistic_cm_tuned$byClass["Specificity"]

logistic_tuned_f1 <- 2 * (
  logistic_tuned_precision * logistic_tuned_recall
) / (
  logistic_tuned_precision + logistic_tuned_recall
)

# ==============================================
# PRINT FINAL RESULTS
# ==============================================

cat("\n=== FINAL TUNED LOGISTIC REGRESSION ===\n")

cat("Selected Threshold:", best_threshold, "\n")

cat("Accuracy:", round(logistic_tuned_accuracy, 4), "\n")

cat("Precision:", round(logistic_tuned_precision, 4), "\n")

cat("Recall:", round(logistic_tuned_recall, 4), "\n")

cat("Specificity:", round(logistic_tuned_specificity, 4), "\n")

cat("F1-Score:", round(logistic_tuned_f1, 4), "\n")

cat("AUC:", round(logistic_auc, 4), "\n")

# ==============================================
# THRESHOLD TUNING VISUALIZATION
# ==============================================

ggplot(logit_threshold_results, aes(x = Threshold)) +
  
  geom_line(aes(y = Precision, color = "Precision"),
            linewidth = 1) +
  
  geom_line(aes(y = Recall, color = "Recall"),
            linewidth = 1) +
  
  geom_line(aes(y = F1_Score, color = "F1-Score"),
            linewidth = 1) +
  
  labs(
    title = "Logistic Regression Threshold Tuning",
    x = "Threshold",
    y = "Metric Value",
    color = "Metrics"
  ) +
  
  theme_bw()

# ==============================================
# PART 13: CROSS VALIDATION
# ==============================================

cat("\n=== 10-FOLD CROSS VALIDATION ===\n")

control <- trainControl(
  method = "cv",
  number = 10
)

set.seed(123)

cv_logistic_model <- train(
  Attrition ~ .,
  data = train_data_balanced,
  method = "glm",
  family = "binomial",
  trControl = control
)

print(cv_logistic_model)

# ==============================================
# PREDICTIONS USING CROSS-VALIDATED MODEL
# ==============================================

cv_pred_prob <- predict(
  cv_logistic_model,
  newdata = test_data,
  type = "prob"
)[,2]

cv_pred_class <- factor(
  ifelse(cv_pred_prob > 0.5, 1, 0),
  levels = c(0,1)
)

cv_cm <- confusionMatrix(
  cv_pred_class,
  test_data$Attrition,
  positive = "1"
)

print(cv_cm)
# ==============================================
# PART 14: MODEL COMPARISON
# ==============================================
cat("\n=== MODEL COMPARISON ===\n")

model_comparison <- data.frame(
  
  Model = c("Logistic Regression",
            "Decision Tree",
            "Random Forest"),
  
  Accuracy = c(
    round(logistic_tuned_accuracy, 4),
    round(decision_tree_accuracy, 4),
    round(rf_base_accuracy, 4)
  ),
  
  Precision = c(
    round(logistic_tuned_precision, 4),
    round(decision_tree_precision, 4),
    round(rf_base_precision, 4)
  ),
  
  Recall = c(
    round(logistic_tuned_recall, 4),
    round(decision_tree_recall, 4),
    round(rf_base_recall, 4)
  ),
  
  F1_Score = c(
    round(logistic_tuned_f1, 4),
    round(decision_tree_f1, 4),
    round(rf_base_f1, 4)
  ),
  
  AUC = c(
    round(logistic_auc, 4),
    round(decision_tree_auc, 4),
    round(rf_base_auc, 4)
  )
)

print(model_comparison)

# ==============================================
# PART 15: ROC CURVES COMPARISON
# ==============================================

cat("\n=== ROC CURVES COMPARISON ===\n")

# Logistic Regression
pred_log <- prediction(logistic_pred_prob, test_data$Attrition)
perf_log <- performance(pred_log, "tpr", "fpr")
auc_log <- performance(pred_log, measure = "auc")@y.values[[1]]

# Decision Tree
pred_dt <- prediction(decision_tree_pred_prob, test_data$Attrition)
perf_dt <- performance(pred_dt, "tpr", "fpr")
auc_dt <- performance(pred_dt, measure = "auc")@y.values[[1]]

# Random Forest
pred_rf <- prediction(rf_base_pred_prob, test_data$Attrition)
perf_rf <- performance(pred_rf, "tpr", "fpr")
auc_rf <- performance(pred_rf, measure = "auc")@y.values[[1]]

# Plot ROC curves
plot(perf_log,
     col = "blue",
     lwd = 2.5,
     main = "ROC Curves - Model Comparison")

plot(perf_dt,
     col = "green",
     lwd = 2.5,
     add = TRUE)

plot(perf_rf,
     col = "red",
     lwd = 2.5,
     add = TRUE)

# Reference diagonal line
abline(a = 0, b = 1,
       lty = 2,
       col = "gray",
       lwd = 2)

# Legend
legend("bottomright",
       legend = c(paste("Logistic Regression (AUC =", round(auc_log, 4), ")"),
                  paste("Decision Tree (AUC =", round(auc_dt, 4), ")"),
                  paste("Random Forest (AUC =", round(auc_rf, 4), ")")),
       col = c("blue", "green", "red"),
       lwd = 2.5)