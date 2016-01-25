library(ggplot2)
library(magrittr)

load("./Qmean_r_long0.rdata")

rm11 <- read.table("./rm11.txt")[[1]]
rm12 <- read.table("./rm12.txt")[[1]]
rm13 <- read.table("./rm13.txt")[[1]]
rm23 <- read.table("./rm23.txt")[[1]]
rm31 <- read.table("./rm31.txt")[[1]]
rm33 <- read.table("./rm33.txt")[[1]]

rmm31 <- sapply(1000:length(rm31), 
                function(i) mean(rm31[(i-999):i]))
rmm23 <- sapply(1000:length(rm23), 
                function(i) mean(rm23[(i-999):i]))
rmm33 <- sapply(1000:length(rm33), 
                function(i) mean(rm33[(i-999):i]))

plot(1:length(rm33), rm33, lty = 1, type = "l", 
     xlim = c(1, 4500), 
     xlab = "Iteration", ylab = "Observed",
     col = "black")




plot(0, 0, type = "n",
     xlim = c(1000, 5000), ylim = c(1, 600), 
     xlab = "Iteration", ylab = "Running Mean Score",
     col = "lightgrey")
lines(1000:length(rm31), rmm31, col = 1)
lines(1000:length(rm23), rmm23, col = 2)
lines(1000:length(rm33), rmm33, col = 3)
legend("topleft", 
       legend = c("50 x 50", "25 x 25", "25 x 10"), 
       lty = 1, col = 1:3)



Qmean_r_long <- 
  lapply(Qmean_r_long, as.numeric) %>% as.data.frame

decn_mean <-
  ggplot(Qmean_r_long, aes(x = x, y = p)) + 
  geom_tile(aes(fill = Qmean), width = 25, height = 10) + 
  scale_fill_gradient(low = "white", high = "red") + 
  xlim(-200, 600) + ylim(-350, 350) +
  labs(x = "Horizontal Difference", y = "Vertical Difference") 

decn_mean

ggsave("../meanDecision.jpg", decn_mean, width = 12, height = 6)
