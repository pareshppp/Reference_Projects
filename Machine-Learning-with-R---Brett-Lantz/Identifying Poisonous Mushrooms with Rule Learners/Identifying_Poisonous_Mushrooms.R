# Identifying poisonous mushrooms with rule learners
setwd(paste0("~/Code/R/R Projects/",
             "Identifying Poisonous Mushrooms/",
             "Identifying_Poisonous_Mushrooms"))

# exploring and preparing the data
mushrooms <- read.csv('mushrooms.csv', stringsAsFactors = T)

mushrooms$type <- factor(x = mushrooms$type, labels = c('edible', 'poisonous'))
mushrooms$cap_shape <- factor(x = mushrooms$cap_shape,
                              labels = c('bell', 'conical', 'flat',
                                         'knobbed', 'sunken', 'convex'))
mushrooms$cap_surface <- factor(x = mushrooms$cap_surface,
                                labels = c('fibrous', 'grooves', 'smooth', 'scaly'))
mushrooms$cap_color <- factor(x = mushrooms$cap_color,
                              labels = c('buff', 'cinnamon', 'red', 'gray', 'brown',
                                         'pink', 'green', 'purple', 'white',
                                         'yellow'))
mushrooms$bruises <- factor(x = mushrooms$bruises, labels = c('no', 'bruises'))
mushrooms$odor <- factor(x = mushrooms$odor, labels = c('almond', 'creosote', 'foul', 
                                                        'anise', 'musty', 'none', 
                                                        'pungent', 'spicy', 'fishy'))
mushrooms$gill_attachment <- factor(x = mushrooms$gill_attachment, 
                                    labels = c('attached', 'free'))
mushrooms$gill_spacing <- factor(x = mushrooms$gill_spacing, 
                                 labels = c('close', 'crowded'))
mushrooms$gill_size <- factor(x = mushrooms$gill_size, labels = c('broad', 'narrow'))
mushrooms$gill_color <- factor(x = mushrooms$gill_color, 
                               labels = c('buff', 'red', 'gray', 'chocolate', 
                                          'black', 'brown', 'orange', 'pink', 
                                          'green', 'purple', 'white', 'yellow'))
mushrooms$stalk_shape <- factor(x = mushrooms$stalk_shape, 
                                labels = c('enlarging', 'tapering'))
mushrooms$stalk_root <- factor(x = mushrooms$stalk_root, 
                               labels = c('missing', 'bulbous', 'club', 'equal', 
                                          'rooted'))
mushrooms$stalk_surface_above_ring <- factor(x = mushrooms$stalk_surface_above_ring, 
                                             labels = c('fibrous', 'silky', 'smooth', 
                                                        'scaly'))
mushrooms$stalk_surface_below_ring <- factor(x = mushrooms$stalk_surface_below_ring, 
                                             labels = c('fibrous', 'silky', 'smooth', 
                                                        'scaly'))
mushrooms$stalk_color_above_ring <- factor(x = mushrooms$stalk_color_above_ring, 
                                           labels = c('buff', 'cinnamon', 'red', 
                                                      'gray', 'brown', 'orange', 
                                                      'pink', 'white', 'yellow'))
mushrooms$stalk_color_below_ring <- factor(x = mushrooms$stalk_color_below_ring, 
                                           labels = c('buff', 'cinnamon', 'red', 
                                                      'gray', 'brown', 'orange', 
                                                      'pink', 'white', 'yellow'))
mushrooms$veil_type <- factor(x = mushrooms$veil_type, labels = c('partial'))
mushrooms$veil_color <- factor(x = mushrooms$veil_color, 
                               labels = c('brown', 'orange', 'white', 'yellow'))
mushrooms$ring_number <- factor(x = mushrooms$ring_number, 
                                labels = c('none', 'one', 'two'))
mushrooms$ring_type <- factor(x = mushrooms$ring_type, 
                              labels = c('evanescent', 'flaring', 'large', 
                                         'none', 'pendant'))
mushrooms$spore_print_color <- factor(x = mushrooms$spore_print_color, 
                                      labels = c('buff', 'chocolate', 'black', 
                                                 'brown', 'orange', 'green', 
                                                 'purple', 'white', 'yellow'))
mushrooms$population <- factor(x = mushrooms$population, 
                               labels = c('abundant', 'clustered', 'numerous', 
                                          'scattered', 'several', 'solitary'))
mushrooms$habitat <- factor(x = mushrooms$habitat, 
                            labels = c('woods', 'grasses', 'leaves', 'meadows', 
                                       'paths', 'urban', 'waste'))

write.csv(x = mushrooms, file = 'mushrooms_labeled.csv')

str(mushrooms)

# removing veil-type as it only has 1 level
mushrooms$veil_type <- NULL

prop.table(table(mushrooms$type))

# For the purposes of this experiment, we will consider the 8,214 samples in the
# mushroom data to be an exhaustive set of all the possible wild mushrooms. This
# is an important assumption, because it means that we do not need to hold some
# samples out of the training data for testing purposes. We are not trying to develop
# rules that cover unforeseen types of mushrooms; we are merely trying to find rules
# that accurately depict the complete set of known mushroom types. Therefore, we can
# build and test the model on the same data.


# training a model on the data - 1R
library(RWeka)

mushrooms.1R <- OneR(formula = type ~ ., data = mushrooms)

# evaluating model performance
mushrooms.1R
summary(mushrooms.1R)

# improving model performance - RIPPER
mushrooms.JRip <- JRip(formula = type ~ ., data = mushrooms)
mushrooms.JRip
