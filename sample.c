/* 
    This program encapsules the entire business logic of the credit card system.
    Data extraction from txt files to extract attributes done

*/ 

#include<stdio.h>
#include<string.h>
#include<stdlib.h>

// Output String conting approval status for all test cases
char* output_string;

//Hosts the functional logic for creadit card approval
//Can be modularized to locate faulty function
void StoreApproval(char attr_uid[],int attr_citizenship,int attr_state,int attr_age,int attr_income,int attr_region,int attr_incomeclass,int attr_maritalstatus,int attr_dependents)
{
	char approved[3]="1\n";
	char unapproved[3]="0\n";
	if(attr_age>21 && (attr_incomeclass==0||attr_incomeclass==1))
		strcat(output_string, unapproved);
	else
		strcat(output_string, approved);
	
}


void main()
{
	output_string = (char*)malloc(1000000*sizeof(char));
	char* attr_uid = (char*)malloc(100*sizeof(char));
	int attr_citizenship,attr_state,attr_age,attr_income,attr_region,attr_incomeclass,attr_maritalstatus,attr_dependents,attr_approval;
	int _countattribute;
	FILE *fptr;

    // Open approval output file to clear contents
	FILE* fptr1;
	fptr1 = fopen("output.txt", "w");
    fclose(fptr1);

    // Open file containing all test cases to be read
    char filename[]="sampledata.txt";
    char ch;
    int cur_len=0;

    //  open the file for reading 
    fptr = fopen(filename, "r");
    if (fptr == NULL)
    {
        printf("Cannot open file \n");
        exit(0);
    }
    ch = fgetc(fptr);
    int flag=0;


    // Logic to extract all the attributes of  particular entity
    while (ch != EOF)
    {
    	if(ch=='\n'||!flag)
    	{

    		//
    		if(flag)
    		{
    			printf("%s %d %d %d %d %d %d %d %d \n",attr_uid,attr_citizenship,attr_state,attr_age,attr_income,attr_region,attr_incomeclass,attr_maritalstatus,attr_dependents);
    			StoreApproval(attr_uid,attr_citizenship,attr_state,attr_age,attr_income,attr_region,attr_incomeclass,attr_maritalstatus,attr_dependents);
    		}
    		flag=1;
    		
    		

    		attr_uid=(char*)malloc(100*sizeof(char));

    		attr_citizenship = 0;
    		attr_state = 0;
    		attr_age = 0;
    		attr_income = 0;
    		attr_region = 0;
    		attr_incomeclass = 0;
    		attr_maritalstatus = 0;
    		attr_dependents = 0;

    		_countattribute = 0;
    	}
    	if(ch!=' ')
    	{
    		switch(_countattribute)
    		{
    			case 0:
    				cur_len = strlen(attr_uid);
					attr_uid[cur_len] = ch;
    				attr_uid[cur_len+1] = '\0';
    				break;
    			case 1:
    				attr_citizenship = (int)ch-48;
    				break;
    			case 2:
    				attr_state = (int)ch-48;
    				break;
    			case 3:
    				attr_age = attr_age*10+((int)ch-48);
    				break;
    			case 4:
    				attr_income = (int)ch-48;
    				break;
    			case 5:
    				attr_region = (int)ch-48;
    				break;
    			case 6:
    				attr_incomeclass = (int)ch-48;
    				break;
    			case 7:
    				attr_maritalstatus = (int)ch-48;
    				break;
    			case 8:
    				attr_dependents = (int)ch-48;
    				break; 
    		}

    	}
    	else
    		_countattribute++;
    	ch = fgetc(fptr);

    }

    fclose(fptr);

    // Write output data to the file
    fptr = fopen("output.txt","w");
    fputs(output_string,fptr);
    fclose(fptr);
	printf("This is working");

}