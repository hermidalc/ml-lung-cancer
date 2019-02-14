#!/usr/bin/env Rscript

options(warn=1)
suppressPackageStartupMessages(library("Biobase"))

dataset_name <- "lhc_nsclc_met"
eset_name <- paste("eset", dataset_name, sep="_")
eset_file <- paste0("data/", eset_name, ".Rda")
cat("Loading:", eset_name, "\n")
load(eset_file)
eset <- get(eset_name)
data <- cbind(t(exprs(eset)), pData(eset))

# neg_126.90693425847_18.6373305017925
mz_colname <- "neg_126.90693425847_18.6373305017925"
cat('M/Z:', mz_colname)
hi_low_cutoff <- quantile(data[data$Class == 0, mz_colname], probs=0.75)
new_data <- data
new_data[[mz_colname]] <- as.factor(ifelse(new_data[, mz_colname] >= hi_low_cutoff, "H", "L"))
fit_glm <- glm(
    Class ~ `neg_126.90693425847_18.6373305017925` + Race + Gender + Smoker,
    data=new_data, family=binomial(link="logit")
)
summary(fit_glm)
exp(cbind(OR=coef(fit_glm), confint(fit_glm)))
cat('\n')
# pos_315.104695148748_132.237212474882
mz_colname <- "pos_315.104695148748_132.237212474882"
cat('M/Z:', mz_colname)
hi_low_cutoff <- quantile(data[data$Class == 0, mz_colname], probs=0.75)
new_data <- data
new_data[[mz_colname]] <- as.factor(ifelse(new_data[, mz_colname] >= hi_low_cutoff, "H", "L"))
fit_glm <- glm(
    Class ~ `pos_315.104695148748_132.237212474882` + Race + Gender + Smoker,
    data=new_data, family=binomial(link="logit")
)
summary(fit_glm)
exp(cbind(OR=coef(fit_glm), confint(fit_glm)))
cat('\n')
# creatine riboside: pos_264.121522378078_23.3092187551771
mz_colname <- "pos_264.121522378078_23.3092187551771"
cat('Creatine riboside:', mz_colname)
hi_low_cutoff <- quantile(data[data$Class == 0, mz_colname], probs=0.75)
new_data <- data
new_data[[mz_colname]] <- as.factor(ifelse(new_data[, mz_colname] >= hi_low_cutoff, "H", "L"))
fit_glm <- glm(
    Class ~ `pos_264.121522378078_23.3092187551771` + Race + Gender + Smoker,
    data=new_data, family=binomial(link="logit")
)
summary(fit_glm)
exp(cbind(OR=coef(fit_glm), confint(fit_glm)))
cat('\n')
