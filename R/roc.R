fileName <- "/home/sudipto/data/projects/lhspred/output/roc/roc_summary.tsv"

data <- read.csv(fileName, sep = "\t")
#print(data)
print(ncol(data))
print(nrow(data))

cols_reg <- c("SVR (AUC=0.9129)", "MLPR (AUC=0.8980)")
x <- as.array(data$FPR)
svr_y <- as.array(data[, 2])
mlpr_y <- as.array(data[, 3])

lwd <- c(1.5, 3)
lty <- c(1, 3)
xlab <- "False positive rate"
ylab <- "True positive rate"
#par(mfrow = c(1, 2))

# svg("roc_plot.svg", width=8, height=8)
plot(x, svr_y, xlab = xlab, ylab = ylab, type = "l", lwd = lwd[1], lty = lty[1], cex.lab=1.2)
#title("a.", adj=0, line=0.5)
lines(x, mlpr_y, type = "l", lwd = lwd[2], lty = lty[2])
legend(x = "bottomright", lty = lty, lwd=lwd, cex=1.1, legend=cols_reg)
# dev.off()

