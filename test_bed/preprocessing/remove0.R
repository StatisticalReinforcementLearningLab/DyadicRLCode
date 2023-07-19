file_location = "test_bed/preprocessing/summary_stats"
file_location_new = "test_bed/preprocessing/summary_stats_removing0"

study_ID = 201
study_directory = sprintf("%s/%s/",file_location,study_ID)
users = list.files(path = study_directory)


missing_count = function(a){sum(is.na(a))/length(a)}
for (user in users){
    filename = sprintf("%s/%s/%s",file_location,study_ID, user)
    load(filename)
    data_one_user[data_one_user == 0] = NA
    filename_new = sprintf("%s/%s/%s",file_location_new,study_ID, user)
    save(data_one_user, file = filename_new)
}

