library(dplyr)
library(geepack)
library(readxl)
pairs = read_excel("Roadmap_IDs/BMT_Roadmap_with_Peds_2023_04_21.xlsx")

file_location = "test_bed/preprocessing/summary_stats_removing0"
impute_file_location = "test_bed/preprocessing/imputed_data"
residual_file_location = "test_bed/preprocessing/residuals_heter"


##
missing_mood_thres = 0.75
missing_other_thres = 0.75
length_thres = 7*15
time_index = 1
missing_count = function(a){sum(is.na(a))/length(a)}

############################
## Aggregate data 
############################
aggregate = function(data0){
  n = dim(data0)[1]
  
  heart_data = cbind(data0[2:n,"morningHEART"], data0[1:(n-1),"dayHEART"], data0[1:(n-1),"nightHEART"])
  heart = apply(heart_data, 1, mean, na.rm = TRUE)
  heart[is.nan(heart)]=NA
  
  mood_data = cbind(data0[1:(n-1),"morningMOOD"], data0[1:(n-1),"dayMOOD"], data0[1:(n-1),"nightMOOD"])
  mood = apply(mood_data, 1, mean, na.rm = TRUE)
  mood[is.nan(mood)]=NA
  
  sleep_data = cbind(data0[2:n,"morningSLEEP"], data0[1:(n-1),"nightSLEEP"])
  sleep = apply(sleep_data, 1, sum, na.rm = TRUE)
  sleep[is.nan(sleep)]=NA
  sleep[sleep==0]=NA
  
  step_data = cbind(data0[2:n,"morningSTEPS"], data0[1:(n-1),"daySTEPS"], data0[1:(n-1),"nightSTEPS"])
  step = apply(step_data, 1, sum, na.rm = TRUE)
  step[is.nan(step)]=NA
  step[step==0]=NA
  
  new_data = data.frame(HEART = heart, MOOD = mood, SLEEP = sleep, STEPS = step)
  return (new_data)
}
############################
## Compliance
############################
compliances = NULL

for (study_ID in c(201,203)){
  study_directory = sprintf("%s/%s/",file_location,study_ID)
  caregivers = list.files(path = study_directory)
  
  
  for (caregiver_number in (1:length(caregivers))){
    compliance = FALSE
    caregiver = substr(caregivers[caregiver_number],1,8)
    if (length(which(pairs[,3]==caregiver)) > 0){
        ## can find a match
        patient = toString(pairs[which(pairs[,3]==caregiver),2])
        filename_caregiver = sprintf("%s/%s/%s.Rda",file_location,study_ID, caregiver)
        load(filename_caregiver)
        data_caregiver0 = data_one_user
        data_caregiver = aggregate(data_caregiver0)
        filename_patient = sprintf("%s/%s/%s.Rda",file_location,202, patient)
        if (file.exists(filename_patient)){
          load(filename_patient)
          data_patient0 = data_one_user
          data_patient = aggregate(data_patient0)
          n = dim(data_patient)[1]
          if (n >= length_thres){

            missingness_caregiver = apply(data_caregiver[time_index:(time_index+length_thres-1),1:4], 2, missing_count) 
            missingness_patient = apply(data_patient[time_index:(time_index+length_thres-1),1:4], 2, missing_count) 
            
            if(
             (missingness_patient[2] < missing_mood_thres) &
             (missingness_caregiver[2] < missing_mood_thres) &
              (max(missingness_patient[c(1,3,4)]) < missing_other_thres)&
             (max(missingness_caregiver[c(1,3,4)]) < missing_other_thres)
            ){
              compliance = TRUE
            }
          }
          compliance_data = data.frame(compliance = compliance,
                                       study_ID = study_ID,
                                       caregiver = caregiver,
                                       patient = patient)
          compliances = rbind(compliances, compliance_data)
        }
    }
  }
}
#sum(compliances)
compliances

complianced_pairs = subset(compliances, compliance)
n_comp = dim(complianced_pairs)[1]
n_comp

############################
## Missing data imputation
############################
## missing data inputation

library(mice)

n = dim(complianced_pairs)[1]
for (pair_no in 1:n){
  print(pair_no)
  caregiver = complianced_pairs[pair_no,3]
  patient = complianced_pairs[pair_no,4]
  study_ID = complianced_pairs[pair_no,2]
  filename_caregiver = sprintf("%s/%s/%s.Rda",file_location,study_ID, caregiver)
  load(filename_caregiver)
  data_caregiver = aggregate(data_one_user)[time_index:(time_index + length_thres),]
  ## save one more day to make sure that "next step" is well defined

  filename_patient = sprintf("%s/%s/%s.Rda",file_location,202, patient)
  load(filename_patient)
  data_patient = aggregate(data_one_user)[time_index:(time_index + length_thres),]

  data_caregiver_imputed = complete(mice(data_caregiver),1)
  data_patient_imputed = complete(mice(data_patient),1)
  new_filename = sprintf("%s/%s.Rda",impute_file_location, patient)

  save(data_caregiver_imputed, data_patient_imputed , file = new_filename)
}




############################
## Analyze data
############################

n = dim(complianced_pairs)[1]
full_data = NULL
full_data_weekly = NULL
for (pair_no in 1:n){
  #for (pair_no in 1){
  print(pair_no)
  caregiver = complianced_pairs[pair_no,3]
  patient = complianced_pairs[pair_no,4]
  study_ID = complianced_pairs[pair_no,2]
  
  filename_imputed = sprintf("%s/%s.Rda",impute_file_location, patient)
  load(filename_imputed)
  data_patient = data_patient_imputed
  data_caregiver = data_caregiver_imputed
  
  weekly_mood_patient0 = apply(matrix(data_patient[1:length_thres,2],nrow = 7), 2, mean, na.rm=TRUE)
  weekly_mood_caregiver0 = apply(matrix(data_caregiver[1:length_thres,2],nrow = 7), 2, mean, na.rm=TRUE)
  weekly_mood_patient = c(rep(NA,7), rep(weekly_mood_patient0[-length(weekly_mood_patient0)], each = 7))
  weekly_mood_caregiver = c(rep(NA,7), rep(weekly_mood_caregiver0[-length(weekly_mood_caregiver0)], each = 7))
  weekly_mood_patient_next = c(rep(weekly_mood_patient0, each = 7))
  weekly_mood_caregiver_next = c(rep(weekly_mood_caregiver0, each = 7))
  data_more = cbind(data_patient[1:length_thres,1:4], data_caregiver[1:length_thres,1:4], 
                    weekly_mood_patient, weekly_mood_caregiver, weekly_mood_patient_next, weekly_mood_caregiver_next)
  #data_more[data_more == "NaN"] <- NA
  
  data_more[,4] = sqrt(data_patient[1:length_thres,4])
  data_more[,8] = sqrt(data_caregiver[1:length_thres,4])
  
  data_patient_next = data_patient[-1,1:4]
  data_patient_next[,4] = sqrt(data_patient_next[,4])
  data_more = cbind(data_more, data_patient_next)
  
  data_caregiver_next = data_caregiver[-1,1:4]
  data_caregiver_next[,4] = sqrt(data_caregiver_next[,4])
  data_more = cbind(data_more, data_caregiver_next)
  
  data_pair_no = rep(pair_no, length_thres)
  data_more = cbind(data_pair_no, data_more)
  
  data_which_group = rep(study_ID==201, length_thres)
  data_more = cbind(data_which_group, data_more)
  
  colnames(data_more) = c("psyc_service","pair_id", "heart", "mood","sleep", "step", 
                          "heart_caregiver", "mood_caregiver","sleep_caregiver", "step_caregiver",
                          "weekly_mood_patient", "weekly_mood_caregiver",
                          "weekly_mood_patient_next", "weekly_mood_caregiver_next",
                          "heart_next", "mood_next","sleep_next", "step_next",
                          "heart_caregiver_next", "mood_caregiver_next","sleep_caregiver_next", "step_caregiver_next")
  data_final = data_more[8:length_thres,]
  full_data = rbind(full_data,data_final)
}

############################
## Standardize
############################
vectors_to_stan = c("heart", "mood","sleep", "step", 
                    "heart_caregiver", "mood_caregiver","sleep_caregiver", "step_caregiver",
                    "weekly_mood_patient", "weekly_mood_caregiver",
                    "weekly_mood_patient_next", "weekly_mood_caregiver_next",
                    "heart_next", "mood_next","sleep_next", "step_next",
                    "heart_caregiver_next", "mood_caregiver_next","sleep_caregiver_next", "step_caregiver_next")
full_data_stan = full_data %>% mutate_at(vectors_to_stan, ~(scale(.) %>% as.vector))

mean_full = apply(full_data[c("heart", "mood","sleep", "step", 
                              "heart_caregiver", "mood_caregiver","sleep_caregiver", "step_caregiver",
                              "weekly_mood_patient", "weekly_mood_caregiver")], 2, mean)
sd_full = apply(full_data[c("heart", "mood","sleep", "step", 
                            "heart_caregiver", "mood_caregiver","sleep_caregiver", "step_caregiver",
                            "weekly_mood_patient", "weekly_mood_caregiver")], 2, sd)
cap_high_original = c(120,10,43200,200,120,10,43200,200,10,10)
cap_low_original = c(55,0,0,0,55,0,0,0,0,0)
cap_high = (cap_high_original-mean_full)/sd_full
cap_low = (cap_low_original-mean_full)/sd_full

filename = sprintf("%s/caps.csv",residual_file_location)
write.csv(rbind(cap_low, cap_high), filename)

## keep track of coefficients
treatment_effects = NULL

############################
## GEE
############################


for (pair_no in 1:n){
  print(pair_no)
  data_final = full_data_stan %>% filter(pair_id == pair_no)
  #### save the original data!
  variables_to_save = c("pair_id", "heart", "sleep", "step", 
                        "heart_caregiver", "sleep_caregiver", "step_caregiver",
                        "weekly_mood_patient", "weekly_mood_caregiver")
  data_original = data_final[1,variables_to_save]
  filename = sprintf("%s/original_pair%s.csv",residual_file_location, pair_no)
  write.csv(data_original, filename)
  
  
  #### get the residuals 
  residual = NULL
  coeffi = NULL
  
  ## patient GEE
  for (y_variable in c("heart","sleep", "step")){
    formula_string = sprintf("%s_next ~ weekly_mood_patient + weekly_mood_caregiver + heart + sleep + step", y_variable)
    formula_touse = formula(formula_string)
    ## analyze data with gee
    fit_gee <- geeglm(formula_touse,
                      data= data_final,
                      id = pair_id, 
                      family = gaussian,
                      corstr = "ar1")
    #print(summary(fit_gee))
    
    ## save residuals
    residual = cbind(residual, fit_gee$residuals)
    coeffi = cbind(coeffi, fit_gee$coefficients)
  }
  treatment_effects = c(treatment_effects,coeffi[6,3])
  
  coeffi_caregiver = NULL
  ## caregiver GEE
  for (y_variable in c("heart_caregiver","sleep_caregiver", "step_caregiver")){
    formula_string = sprintf("%s_next ~ weekly_mood_patient + weekly_mood_caregiver + heart_caregiver + sleep_caregiver + step_caregiver", y_variable)
    formula_touse = formula(formula_string)
    ## analyze data with gee
    fit_gee <- geeglm(formula_touse,
                      data= data_final,
                      id = pair_id, 
                      family = gaussian,
                      corstr = "ar1")
    residual = cbind(residual, fit_gee$residuals)
    coeffi_caregiver = cbind(coeffi_caregiver, fit_gee$coefficients)
  }
  residual = data.frame(residual)
  colnames(residual) = c("heart", "sleep", "step", "heart_caregiver", "sleep_caregiver", "step_caregiver")
  coeffi = data.frame(coeffi)
  colnames(coeffi) = c("heart", "sleep", "step")
  coeffi_caregiver = data.frame(coeffi_caregiver)
  colnames(coeffi_caregiver) = c("heart_caregiver", "sleep_caregiver", "step_caregiver")
  
  filename = sprintf("%s/residual_pair%s.csv",residual_file_location, pair_no)
  write.csv(residual, filename)
  filename = sprintf("%s/coeffi_pair%s.csv",residual_file_location, pair_no)
  write.csv(coeffi, filename)
  filename = sprintf("%s/coeffi_caregiver_pair%s.csv",residual_file_location, pair_no)
  write.csv(coeffi_caregiver, filename)
  
  
  ############################
  #analyzing the weekly mood change
  residual_weekly = NULL
  coeffi_weekly = NULL
  data_sunday = data_final[(1:14)*7,c("weekly_mood_patient","weekly_mood_caregiver","weekly_mood_patient_next","weekly_mood_caregiver_next", "pair_id")]
  fit_gee <- geeglm(weekly_mood_patient_next~weekly_mood_patient+weekly_mood_caregiver,
                    data = data_sunday,
                    id = pair_id, 
                    family = gaussian,
                    corstr = "ar1")
  residual_weekly = cbind(residual_weekly, fit_gee$residuals)
  coeffi_weekly = cbind(coeffi_weekly, fit_gee$coefficients)
  
  fit_gee <- geeglm(weekly_mood_caregiver_next~weekly_mood_patient+weekly_mood_caregiver,
                    data = data_sunday,
                    id = pair_id, 
                    family = gaussian,
                    corstr = "ar1")
  residual_weekly = cbind(residual_weekly, fit_gee$residuals)
  coeffi_weekly = cbind(coeffi_weekly, fit_gee$coefficients)
  
  filename = sprintf("%s/residual_weekly_pair%s.csv",residual_file_location, pair_no)
  write.csv(residual_weekly, filename)
  filename = sprintf("%s/coeffi_weekly_pair%s.csv",residual_file_location, pair_no)
  write.csv(coeffi_weekly, filename)
  
}

