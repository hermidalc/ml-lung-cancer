#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("argparse"))

# parser <- ArgumentParser()
# args <- parser$parse_args()

dataset_tr_name <- "lhc_nsclc_met"
suffixes <- c()
df_x_tr_name <- paste0(c("df", "X", dataset_tr_name, suffixes), collapse="_")
df_p_tr_name <- paste0(c("df", "p", dataset_tr_name, suffixes), collapse="_")
cat("Creating:", df_x_tr_name, "+", df_p_tr_name, "\n")
df_x_tr <- read.delim(paste0("data/", dataset_tr_name, "_data.txt"), row.names=1)
df_p_tr <- read.delim(paste0("data/", dataset_tr_name, "_meta.txt"), row.names=2)
assign(df_x_tr_name, df_x_tr)
assign(df_p_tr_name, df_p_tr)
save(list=df_x_tr_name, file=paste0("data/", df_x_tr_name, ".Rda"))
save(list=df_p_tr_name, file=paste0("data/", df_p_tr_name, ".Rda"))
