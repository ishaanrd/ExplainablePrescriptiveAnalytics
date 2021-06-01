library(bupaR)

log <- read.csv("./PyOut/log4bupar.csv")

df <- log

df[,'timestamp'] <- as.POSIXct(df[,'timestamp'], origin = "2020-01-01", tz = "GMT")

elog <- eventlog(eventlog = df, 
                 case_id = "case_id",
                 activity_id = "event",
                 activity_instance_id = "event_instance",
                 lifecycle_id = "status",
                 timestamp = "timestamp",
                 resource_id = "resource")
elog %>%
  end_activities("activity")

elog %>%
  start_activities("activity")

elog %>%
  process_map(performance(mean, "mins"))

elog %>% filter_activity_frequency(percentage = 0.75) %>% process_map()

#filtering the log on endpoints of Registered and finished
felog <- filter_endpoints(elog, start_activities = "registered", end_activities = "finished")

length(unique(felog$case_id))

felog %>% process_map(type_nodes = frequency("relative_case"),
                      type_edges = performance(mean))

animate_process(felog, mapping = token_aes(color = token_scale("red")),
                type_nodes = frequency("relative_case"),
                type_edges = performance(mean))
