variables = c("HEART", "MOOD", "SLEEP", "STEPS")

ranges = list(c(0,8),c(8,20),c(20,24))

# "HEART", "MOOD", "SLEEP", "STEPS"

turbo_link = ## link_to_data

start_time = Sys.time()

study_ID = 201
study_directory = sprintf("%s/%s/",turbo_link,study_ID)
users = list.files(path = study_directory)

compute_summary = function(data_this_day, range, variable){
    start_hour = as.numeric(format(as.POSIXct(as.POSIXct(data_this_day[,1])), format = "%H"))
    end_hour = as.numeric(format(as.POSIXct(as.POSIXct(data_this_day[,2])), format = "%H"))
    start_minute = as.numeric(format(as.POSIXct(as.POSIXct(data_this_day[,1])), format = "%M"))
    end_minute = pmax(as.numeric(format(as.POSIXct(as.POSIXct(data_this_day[,2])), format = "%M")),
                      start_minute+1)
    start_time = start_hour + start_minute/60
    end_time = end_hour + end_minute/60
    
    overlap = pmin(range[2],end_time) - pmax(range[1],start_time)
    
    proportion_within_range = (overlap>1/60/2)*overlap/(end_time - start_time)
    
    if(variable == "HEART" | variable == "MOOD"){
        if(sum(proportion_within_range) == 0){
            return(NA)
        }
        result = sum(proportion_within_range*data_this_day[,3])/sum(proportion_within_range)
        if(variable == "MOOD" & result == 0){
            return(NA)
        }
        return(result)
    } else if(variable =="SLEEP"){
        return(sum(proportion_within_range*data_this_day[,3]))
    } else{
        return(sum(proportion_within_range*(data_this_day[,3]-1)))
    }
}

for (user in users[1:86]){
    ## get a data file for one patient
    #user = "YFTMNY6X"
    print(which(user == users))
    directory = sprintf("%s/%s/%s/",turbo_link, study_ID,user)
    dates = list.files(path = directory)
    matrix_one_user = NULL
    
    for (date in dates){
        dir_date = sprintf("%s/%s/%s/%s", turbo_link, study_ID, user, date)
        variable_one_day = NULL
        for (variable_no in 1:4){
            files = list.files(dir_date,pattern=variables[variable_no])
            if (length(files) == 0){
                variable_entry = rep(NA,3)
            } else{
                dir_file = sprintf("%s/%s/%s/%s/%s", turbo_link, study_ID, user, date, files[1])
                data_this_day = read.csv(dir_file,header=FALSE)
                variable_entry = NULL
                for (range in ranges){
                    variable_entry = c(variable_entry,compute_summary(data_this_day[,1:3], range, variables[variable_no]))
                }
            }
            variable_one_day = c(variable_one_day, variable_entry)
        }
        matrix_one_user = rbind(matrix_one_user, variable_one_day)
    }
    data_one_user = as.data.frame(matrix_one_user)
    colnames(data_one_user) = c(outer( c("morning","day","night"), variables, FUN=paste0))
    data_one_user[,"date"] = dates
    #print(data_one_user)
    
    output_dir = sprintf("test_bed/preprocessing/summary_stats/%s/%s.Rda", study_ID, user)
    
    save(data_one_user,file=output_dir)
}

end_time = Sys.time()
print(end_time - start_time)
