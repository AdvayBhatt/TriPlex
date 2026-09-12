# Variance components for C1 yield -- the analysis the original script was aiming at.
#
# Why the original formulation cannot run: `environment` (1,177 levels) and
# `population` (499) as FIXED effects build a dense design matrix of roughly
# 516,000 x 1,676 doubles, which exceeds R's 16 GB vector limit. Random effects use
# sparse matrices, so the same model fits comfortably.
#
# It is also the statistically correct choice: VarCorr() reports variance components,
# and a term must be RANDOM to have a variance component at all.
suppressPackageStartupMessages({library(readr); library(lme4)})

data_dir <- Sys.getenv("DATA_DIR", "data/raw")
ph <- read_csv(file.path(data_dir, "C1_Phenotype_Data_V2.csv"),
               show_col_types = FALSE, progress = FALSE)

d <- data.frame(
  yield       = ph$YLD_BE,
  population  = factor(ph$shorthand_x),
  environment = factor(paste(ph$LOC, ph$YEAR_x, sep = "_"))
)
d <- droplevels(d[complete.cases(d), ])
cat("rows", nrow(d), "| populations", nlevels(d$population),
    "| environments", nlevels(d$environment), "\n\n")

# tester is omitted: every population uses exactly one, so it is perfectly nested
# and carries no information separable from population.
m <- lmer(yield ~ 1 + (1 | environment) + (1 | population) +
            (1 | population:environment),
          data = d, REML = TRUE,
          control = lmerControl(optimizer = "bobyqa",
                                optCtrl = list(maxfun = 2e5)))

vc <- as.data.frame(VarCorr(m))
vc$pct <- round(100 * vc$vcov / sum(vc$vcov), 1)
print(vc[, c("grp", "vcov", "sdcor", "pct")])
