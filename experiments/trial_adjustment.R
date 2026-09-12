# Historical-only nuisance adjustment. Genomic prediction is fitted separately.
suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1], stringsAsFactors=FALSE)
stopifnot(max(d$year) < as.integer(args[3]))
for (v in c('year','generation','environment','population','line','tester')) d[[v]] <- factor(d[[v]])
base <- yield ~ year + generation + (1|environment) + (1|population) + (1|line) + (1|population:environment)
if (nlevels(d$generation) < 2) base <- update(base, . ~ . - generation)
if (nlevels(d$year) < 2) base <- update(base, . ~ . - year)
for (variant in c('joint','joint_tester')) {
  formula <- if (variant == 'joint') base else update(base, . ~ . + (1|tester))
  cat('Fitting', variant, nrow(d), 'historical line-site means\n'); flush.console()
  fit <- lmer(formula, d, REML=TRUE, control=lmerControl(optimizer='bobyqa',
              calc.derivs=FALSE, optCtrl=list(maxfun=10000)))
  # Retain the family and line signal AND the line-mean residual. Do not train
  # a ridge on already-shrunken line BLUPs alone (which would shrink twice).
  nuisance <- predict(fit, re.form=if (variant == 'joint')
                        ~(1|environment)+(1|population:environment) else
                        ~(1|environment)+(1|population:environment)+(1|tester))
  adjusted <- aggregate(d$yield-nuisance, list(LINE_UNIQUE_ID=as.character(d$line)), mean)
  names(adjusted)[2] <- 'y'
  write.csv(adjusted, paste0(args[2], '_', variant, '_targets.csv'), row.names=FALSE)
  vc <- as.data.frame(VarCorr(fit))
  report <- list(formula=paste(deparse(formula),collapse=' '), n=nrow(d),
                 max_training_year=max(as.integer(as.character(d$year))),
                 forecast_year=as.integer(args[3]), variance_components=vc,
                 fixed_effects=as.list(fixef(fit)), singular=isSingular(fit),
                 optimizer_code=fit@optinfo$conv$opt,
                 messages=fit@optinfo$conv$lme4$messages,
                 derivatives_checked=FALSE, REML_criterion=REMLcrit(fit),
                 runtime=list(R=R.version.string,lme4=as.character(packageVersion('lme4'))))
  write_json(report, paste0(args[2], '_', variant, '_fit.json'), pretty=TRUE, auto_unbox=TRUE)
  print(vc[,c('grp','vcov')]); flush.console()
  rm(fit); gc()
}
