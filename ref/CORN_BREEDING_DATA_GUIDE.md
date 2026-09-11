# Corn Breeding Program Dataset Guide (2001-2008)

## Overview
This dataset contains comprehensive corn breeding program data spanning from 2001 to 2008, organized into phenotypic, environmental, and genomic components across two genetic clusters (C1 and C2). The data represents a multi-location, multi-year breeding program with extensive phenotypic measurements, environmental characterization, and genomic marker information.

## Dataset Components

### 1. Phenotypic Data
- **C1_Phenotype_Data_V2.csv** (141 MB): Phenotypic measurements for Cluster 1 individuals
- **C2_Phenotype_Data_V2.csv** (141 MB): Phenotypic measurements for Cluster 2 individuals

**Key Columns:**
- `YEAR`: Growing season year (2001-2008)
- `LOC`: Location code (e.g., NEDA, IAPR, IADA, ILBM, ILMN, INVI, MOBU)
- `LINE`: Individual plant/line identifier within populations
- `LONGITUDE`, `LATITUDE`: Geographic coordinates
- `CLUSTER`: Genetic cluster assignment (1 or 2)
- **Phenotypic Traits:**
  - `ERM`: Estimated relative maturity
  - `MST`: Moisture content
  - `PHT`: Plant height
  - `RTLP`: Root lodging percentage
  - `STLP`: Stalk lodging percentage
  - `TWT`: Test weight
  - `YLD_BE`: Yield (bushels per acre)
  - `EHT`: Ear height
  - `SET`: Set ID

### 2. Environmental Data
- **environmental_features.csv** (217 KB): Environmental conditions by year and location

**Key Columns:**
- `YEAR`: Growing season year (2001-2008)
- `LOC`: Location code (matching phenotype data)
- **Weather Variables:**
  - `PRCP_*`: Precipitation measurements (various periods)
  - `TMAX_*`, `TMIN_*`: Maximum and minimum temperatures
  - `TAVG_*`: Average temperatures
- **Soil Characteristics:**
  - `clay_pct`, `silt_pct`, `sand_pct`: Soil texture percentages
  - `ph_h2o`: Soil pH
  - `n_pct`: Nitrogen percentage

### 3. Genomic Data

#### Cluster 1 (C1)
- **ImputedPopulationsC1/**: Folder containing imputed genomic data for 190+ populations
- **Files**: C1.1_Imputed.csv through C1.190+_Imputed.csv

#### Cluster 2 (C2)
- **ImputedC2Populations.zip**: Compressed folder containing imputed genomic data for C2 populations
- **Content**: Similar structure to C1 with C2.1_Imputed.csv through C2.100+_Imputed.csv

**Genomic Data Format:**
- **First column**: Individual identifiers (e.g., "PID200761", "PID1589589", "00000000001", "00000000002")
- **First two rows**: Parent lines of the population
- **Remaining rows**: Progeny lines numbered sequentially (e.g., "00000000001" = line 1, "00000000002" = line 2)
- **Columns**: SNP marker names (e.g., "M00003409443", "M00000005000")
- **Values**: Genotype calls (1, 0, -1 representing different allelic states, NA for missing data)
- **Data consistency**: Genotype coding (1, 0, -1) is consistent across all populations

### 4. Documentation
- **Data Dictionary.docx**: Detailed variable definitions and measurement protocols
- **Data Overview.pptx**: Visual summary of the dataset structure

### 5. Raw Data Archives
- **Unimputed_C1_Genome_Data.zip**: Original, unprocessed genomic data for Cluster 1
- **Unimputed_C2_Genome_Data.zip**: Original, unprocessed genomic data for Cluster 2

## Data Relationships

### Primary Linking Keys

1. **Phenotype ↔ Environment**: `YEAR` + `LOC`
2. **Phenotype ↔ Genomic**: `LINE_UNIQUE_ID` formatted as "Cluster.Population.Line" (e.g., "C1.1.1" for Cluster 1, Population 1, Line 1)
3. **Geographic**: `LONGITUDE` + `LATITUDE` coordinates

### ID Matching System

The `LINE_UNIQUE_ID` in phenotype data follows the pattern:
- **Format**: `C{Cluster}.{Population}.{Line}`
- **Examples**: 
  - `C1.1.1` = Cluster 1, Population 1, Line 1
  - `C2.50.15` = Cluster 2, Population 50, Line 15

This directly corresponds to:
- **Genomic files**: `C1.1_Imputed.csv` contains individuals that match `C1.1.*` phenotype IDs
- **Individual matching**: 
  - **Parent lines**: First two rows in genomic files (e.g., "PID200761", "PID1589589")
  - **Progeny lines**: Numbered sequentially as "00000000001" (line 1), "00000000002" (line 2), etc.
  - **Matching rule**: Extract line number from `LINE_UNIQUE_ID` third component and convert genomic IDs to integers by stripping leading zeros

### Hierarchical Structure
```
Year-Location Combination
├── Environmental conditions (one record per year-location)
├── Multiple individuals tested
│   ├── Phenotypic measurements
│   └── Genomic profiles (if available)
```

## Data Merging Examples

### R Examples

#### 1. Merge Phenotype with Environmental Data
```r
# Load required libraries
library(dplyr)

# Read the data
phenotype_c1 <- read.csv("C1_Phenotype_Data_V2.csv")
environmental <- read.csv("environmental_features.csv")

# Merge phenotype with environmental data
pheno_env_c1 <- phenotype_c1 %>%
  left_join(environmental, by = c("YEAR", "LOC"))

# View the result
head(pheno_env_c1[, c("YEAR", "LOC", "YLD_BE", "PRCP_1", "TAVG_1")])
```

#### 2. Combine C1 and C2 Phenotype Data
```r
# Read both phenotype datasets
phenotype_c1 <- read.csv("C1_Phenotype_Data_V2.csv")
phenotype_c2 <- read.csv("C2_Phenotype_Data_V2.csv")

# Combine both clusters
combined_phenotypes <- rbind(
  phenotype_c1 %>% select(YEAR, LOC, LINE, CLUSTER, YLD_BE, PHT, MST),
  phenotype_c2 %>% select(YEAR, LOC, LINE, CLUSTER, YLD_BE, PHT, MST)
)

# Summarize by cluster
cluster_summary <- combined_phenotypes %>%
  group_by(CLUSTER) %>%
  summarise(
    mean_yield = mean(YLD_BE, na.rm = TRUE),
    mean_height = mean(PHT, na.rm = TRUE),
    n_observations = n()
  )
```

#### 3. Merge Phenotype with Genomic Data
```r
# Read phenotype data
phenotype_c1 <- read.csv("C1_Phenotype_Data_V2.csv")

# Example: Work with population 1 from cluster 1
# Read genomic data for population 1
genomic_c1_pop1 <- read.csv("ImputedPopulationsC1/C1.1_Imputed.csv", row.names = 1)

# Filter phenotype data for population 1 (LINE_UNIQUE_ID starting with "C1.1.")
pheno_pop1 <- phenotype_c1 %>%
  filter(grepl("^C1\\.1\\.", LINE_UNIQUE_ID))

# Extract line numbers from LINE_UNIQUE_ID (the third part: C1.1.X)
pheno_pop1$line_number <- as.numeric(sub("^C1\\.1\\.", "", pheno_pop1$LINE_UNIQUE_ID))

# Get available individuals from genomic data (excluding parent rows)
genomic_individuals <- rownames(genomic_c1_pop1)
progeny_individuals <- genomic_individuals[grepl("^\\d{11}$", genomic_individuals)]

# Create a mapping from genomic IDs to line numbers by stripping leading zeros
genomic_progeny <- genomic_c1_pop1[progeny_individuals, ]
genomic_progeny$line_number <- as.numeric(progeny_individuals)

# Merge phenotype and genomic data by line number
pheno_genomic <- pheno_pop1 %>%
  left_join(
    genomic_progeny[, 1:10] %>% # First 10 markers for example
      rownames_to_column("genomic_id") %>%
      mutate(line_number = as.numeric(genomic_id)),
    by = "line_number"
  )

# View the merged data
head(pheno_genomic[, c("LINE_UNIQUE_ID", "YLD_BE", "PHT", names(genomic_progeny)[1:3])])
```

#### 4. Function for Automated Population Matching
```r
# Function to merge any population's phenotype and genomic data
merge_population_data <- function(cluster, population, phenotype_data) {
  
  # Create pattern for LINE_UNIQUE_ID
  pattern <- paste0("^C", cluster, "\\.", population, "\\.")
  
  # Filter phenotype data for this population
  pheno_subset <- phenotype_data %>%
    filter(grepl(pattern, LINE_UNIQUE_ID)) %>%
    mutate(line_number = as.numeric(sub(pattern, "", LINE_UNIQUE_ID)))
  
  # Read corresponding genomic file
  genomic_file <- paste0("ImputedPopulationsC", cluster, "/C", cluster, ".", 
                        population, "_Imputed.csv")
  
  if(file.exists(genomic_file)) {
    genomic_data <- read.csv(genomic_file, row.names = 1)
    
    # Get progeny individuals only (exclude parents)
    genomic_individuals <- rownames(genomic_data)
    progeny_individuals <- genomic_individuals[grepl("^\\d{11}$", genomic_individuals)]
    genomic_progeny <- genomic_data[progeny_individuals, ]
    
    # Create line numbers by stripping leading zeros from genomic IDs
    genomic_with_lines <- genomic_progeny %>% 
      rownames_to_column("genomic_id") %>%
      mutate(line_number = as.numeric(genomic_id))
    
    # Merge data
    merged_data <- pheno_subset %>%
      left_join(genomic_with_lines, by = "line_number")
    
    return(merged_data)
  } else {
    warning(paste("Genomic file not found:", genomic_file))
    return(pheno_subset)
  }
}

# Example usage:
# merged_c1_pop1 <- merge_population_data(1, 1, phenotype_c1)
# merged_c1_pop5 <- merge_population_data(1, 5, phenotype_c1)
```

### Python Examples

#### 1. Merge Phenotype with Environmental Data
```python
import pandas as pd
import numpy as np

# Read the data
phenotype_c1 = pd.read_csv("C1_Phenotype_Data_V2.csv")
environmental = pd.read_csv("environmental_features.csv")

# Merge phenotype with environmental data
pheno_env_c1 = phenotype_c1.merge(
    environmental, 
    on=['YEAR', 'LOC'], 
    how='left'
)

# View the result
print(pheno_env_c1[['YEAR', 'LOC', 'YLD_BE', 'PRCP_1', 'TAVG_1']].head())
```

#### 2. Combine C1 and C2 Phenotype Data
```python
# Read both phenotype datasets
phenotype_c1 = pd.read_csv("C1_Phenotype_Data_V2.csv")
phenotype_c2 = pd.read_csv("C2_Phenotype_Data_V2.csv")

# Select common columns
common_cols = ['YEAR', 'LOC', 'LINE', 'CLUSTER', 'YLD_BE', 'PHT', 'MST']

# Combine both clusters
combined_phenotypes = pd.concat([
    phenotype_c1[common_cols],
    phenotype_c2[common_cols]
], ignore_index=True)

# Summarize by cluster
cluster_summary = combined_phenotypes.groupby('CLUSTER').agg({
    'YLD_BE': ['mean', 'count'],
    'PHT': 'mean',
    'MST': 'mean'
}).round(2)

print(cluster_summary)
```

#### 3. Merge Phenotype with Genomic Data
```python
import pandas as pd
import re

# Read phenotype data
phenotype_c1 = pd.read_csv("C1_Phenotype_Data_V2.csv")

# Example: Work with population 1 from cluster 1
# Filter phenotype data for population 1
pheno_pop1 = phenotype_c1[phenotype_c1['LINE_UNIQUE_ID'].str.match(r'^C1\.1\.')]

# Extract line numbers from LINE_UNIQUE_ID
pheno_pop1 = pheno_pop1.copy()
pheno_pop1['line_number'] = pheno_pop1['LINE_UNIQUE_ID'].str.extract(r'^C1\.1\.(\d+)').astype(int)

# Read genomic data for population 1
genomic_c1_pop1 = pd.read_csv("ImputedPopulationsC1/C1.1_Imputed.csv", index_col=0)

# Get progeny individuals only (exclude parents - first two rows)
genomic_individuals = list(genomic_c1_pop1.index)
progeny_individuals = [ind for ind in genomic_individuals if re.match(r'^\d{11}$', ind)]

# Filter genomic data for progeny and convert genomic IDs to line numbers
genomic_progeny = genomic_c1_pop1.loc[progeny_individuals]

# Select first 10 markers for example and reset index
genomic_subset = genomic_progeny.iloc[:, :10].reset_index()
genomic_subset.rename(columns={'index': 'genomic_id'}, inplace=True)
# Convert genomic IDs to line numbers by stripping leading zeros
genomic_subset['line_number'] = genomic_subset['genomic_id'].astype(int)

# Merge phenotype and genomic data by line number
pheno_genomic = pheno_pop1.merge(
    genomic_subset.drop('genomic_id', axis=1), 
    on='line_number', 
    how='left'
)

print(f"Merged dataset shape: {pheno_genomic.shape}")
print(pheno_genomic[['LINE_UNIQUE_ID', 'YLD_BE', 'PHT'] + list(genomic_subset.columns[1:4])].head())
```

#### 4. Function for Automated Population Matching
```python
def merge_population_data(cluster, population, phenotype_data):
    """
    Merge phenotype and genomic data for a specific cluster and population
    
    Args:
        cluster (int): Cluster number (1 or 2)
        population (int): Population number
        phenotype_data (DataFrame): Phenotype dataset
    
    Returns:
        DataFrame: Merged phenotype and genomic data
    """
    
    # Create pattern for LINE_UNIQUE_ID
    pattern = f'^C{cluster}\\.{population}\\.'
    
    # Filter phenotype data for this population
    pheno_subset = phenotype_data[phenotype_data['LINE_UNIQUE_ID'].str.match(pattern)]
    
    if pheno_subset.empty:
        print(f"No phenotype data found for C{cluster}.{population}")
        return pheno_subset
    
    # Extract line numbers and create genomic IDs
    pheno_subset = pheno_subset.copy()
    pheno_subset['line_number'] = pheno_subset['LINE_UNIQUE_ID'].str.extract(
        f'^C{cluster}\\.{population}\\.(\\d+)'
    ).astype(int)
    
    # Read corresponding genomic file
    genomic_file = f"ImputedPopulationsC{cluster}/C{cluster}.{population}_Imputed.csv"
    
    try:
        genomic_data = pd.read_csv(genomic_file, index_col=0)
        
        # Get progeny individuals only (exclude parents)
        genomic_individuals = list(genomic_data.index)
        progeny_individuals = [ind for ind in genomic_individuals if re.match(r'^\\d{11}$', ind)]
        
        if not progeny_individuals:
            print(f"No progeny individuals found in genomic data for C{cluster}.{population}")
            return pheno_subset
        
        # Filter genomic data for progeny and convert IDs to line numbers
        genomic_progeny = genomic_data.loc[progeny_individuals].reset_index()
        genomic_progeny.rename(columns={'index': 'genomic_id'}, inplace=True)
        # Convert genomic IDs to line numbers by stripping leading zeros
        genomic_progeny['line_number'] = genomic_progeny['genomic_id'].astype(int)
        
        # Merge data by line number
        merged_data = pheno_subset.merge(
            genomic_progeny.drop('genomic_id', axis=1), 
            on='line_number', 
            how='left'
        )
        
        print(f"Successfully merged C{cluster}.{population}: {merged_data.shape[0]} individuals, {merged_data.shape[1]} total columns")
        return merged_data
        
    except FileNotFoundError:
        print(f"Genomic file not found: {genomic_file}")
        return pheno_subset
    except Exception as e:
        print(f"Error processing C{cluster}.{population}: {e}")
        return pheno_subset

# Example usage:
# merged_c1_pop1 = merge_population_data(1, 1, phenotype_c1)
# merged_c1_pop5 = merge_population_data(1, 5, phenotype_c1)

# Process multiple populations
def process_multiple_populations(cluster, populations, phenotype_data):
    """Process multiple populations and return combined dataset"""
    all_merged = []
    
    for pop in populations:
        merged = merge_population_data(cluster, pop, phenotype_data)
        if not merged.empty:
            all_merged.append(merged)
    
    if all_merged:
        return pd.concat(all_merged, ignore_index=True)
    else:
        return pd.DataFrame()

# Example: Process first 5 populations of cluster 1
# combined_data = process_multiple_populations(1, [1, 2, 3, 4, 5], phenotype_c1)
```

## Analysis Recommendations

### Getting Started Steps

1. **Data Exploration**
   - Start with environmental and phenotype data (smaller files)
   - Understand the year-location structure
   - Examine trait distributions and missing data patterns

2. **Quality Control**
   - Check for missing values in key linking columns (YEAR, LOC)
   - Validate coordinate consistency across years
   - Assess genomic data completeness

3. **Analysis Approaches**
   - **Environmental Analysis**: Characterize growing conditions across locations and years
   - **Phenotypic Analysis**: Compare traits between clusters and environments
   - **Genomic Analysis**: Population structure, diversity, and marker-trait associations
   - **Integrated Analysis**: Genotype × Environment interactions

### Advanced Merging Considerations

1. **Individual ID Matching**: The `LINE_UNIQUE_ID` follows the pattern `C{Cluster}.{Population}.{Line}` which maps to genomic files as follows:
   - **Genomic files**: `C{Cluster}.{Population}_Imputed.csv`  
   - **Parent lines**: First two rows in genomic files (e.g., "PID200761", "PID1589589")
   - **Progeny lines**: Sequential numbering as "00000000001", "00000000002", etc.
   - **Matching strategy**: Extract line number from `LINE_UNIQUE_ID` and convert genomic IDs to integers by removing leading zeros

2. **Population Structure**: Each cluster contains multiple populations, with genomic data organized by population files. Students should:
   - Extract cluster and population numbers from `LINE_UNIQUE_ID`
   - Match to the corresponding genomic file
   - Use proper individual ID formatting for linking

3. **Missing Data**: Some considerations for data integration:
   - Not all phenotyped individuals may have genomic data
   - Some genomic individuals may lack phenotypic measurements
   - Parent lines in genomic data may not have corresponding phenotype records
   - Use appropriate join types (left, inner, outer) based on analysis goals

4. **Computational Resources**: 
   - Process populations individually to manage memory usage
   - Consider data types and efficient storage formats
   - Use chunked processing for large-scale analyses
   - Be aware that genomic files contain both parents and progeny

## File Size Considerations

- **Large Files**: Phenotype CSVs (~141 MB each) may require chunked reading for some analyses
- **Genomic Data**: Individual population files are manageable (~1-1.4 MB each)
- **Memory Management**: Consider data types (factors vs. characters) and subsetting strategies


