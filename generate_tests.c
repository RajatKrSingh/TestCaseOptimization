/* This program generates all the test cases for out usecase*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

// Converts string to int
char* itoa(int val, int base)
{	
	static char buf[32] = {0};

	if(val==0)
	{
		char* bugf =(char*) malloc(10*sizeof(char));
		bugf[0]='0';
		bugf[1]='\0';
		return bugf;
	}
	int i = 30;
	
	for(; val && i ; --i, val /= base)
	
		buf[i] = "0123456789abcdef"[val % base];
	
	return &buf[i+1];
	
}

//Generate all combinations of test cases
//Logic can be implemented on top of this to reduce number of test cases arbitrarily
void main()
{
	int count = 0;
	FILE* fptr1;
	fptr1 = fopen("sampledata.txt", "w");
    fclose(fptr1);
	for(int i_citizenship=0; i_citizenship<=1;i_citizenship++)
	{
		for(int i_state=0;i_state<=1;i_state++)
		{
			for(int i_age=0;i_age<=100;i_age+=1)
			{
				for(int i_income=0;i_income<=1;i_income++)
				{
					for(int i_region=0;i_region<=6;i_region++)
					{
						for(int i_incomeclass=0;i_incomeclass<=3;i_incomeclass++)
						{
							for(int i_maritalstatus=0;i_maritalstatus<=1;i_maritalstatus++)
							{
								for(int i_depedents=0;i_depedents<=4;i_depedents++)
								{
									char testline[100000] = "";
									
									char* scount = (char*)malloc(10000^sizeof(char));
									scount = itoa(count,10);
									for(int i_uid=0;i_uid<(7-strlen(scount));i_uid++)
									{
										strcat(testline,"0");
									}
									strcat(testline,scount);
									strcat(testline," ");
									strcat(testline,itoa(i_citizenship,10));
									strcat(testline," ");
									strcat(testline,itoa(i_state,10));
									strcat(testline," ");
									strcat(testline,itoa(i_age,10));
									strcat(testline," ");
									strcat(testline,itoa(i_income,10));
									strcat(testline," ");
									strcat(testline,itoa(i_region,10));
									strcat(testline," ");
									strcat(testline,itoa(i_incomeclass,10));
									strcat(testline," ");
									strcat(testline,itoa(i_maritalstatus,10));
									strcat(testline," ");
									strcat(testline,itoa(i_depedents,10));

									strcat(testline,"\n");
									printf("%s",testline);
									
									//testline = testline + " " + i_citizenship + " " + i_state + " " + i_age + " " + i_income + " " + i_region + " " + i_incomeclass + " " + i_maritalstatus + " " + i_depedents;
									FILE* fptr;

									fptr = fopen("sampledata.txt", "a");
    								fputs(testline, fptr);
    								fclose(fptr);
    								count++;

								}
							}
						}
					}
				}
			}
		}
	}
}