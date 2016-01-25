library(ggplot2)
library(magrittr)

load("./Qmean_r_long.rdata")

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
