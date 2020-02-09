import os, sys
import numpy
import sysconfig
import urllib2
import util

module_path=''
chkparseC="""
/* On the first usage this gets printed in a .c file which is authomatically compiled */
/* content of the chkparse C script used to make the checkpoint file from psiblast more user friendly */
/* chkparse - generate PSIPRED compatible mtx file from BLAST+ checkpoint file */
/* V0.3 */
/* Copyright (C) 2010 D.T. Jones */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#define MAXSEQLEN 65536
#define EPSILON 1e-6
#define FALSE 0
#define TRUE 1
#define SQR(x) ((x)*(x))
#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))
const char *ncbicodes = "*A*CDEFGHIKLMNPQRSTVWXY*****";
/*  BLOSUM 62 */
const short           aamat[23][23] =
{
    {4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0},
    {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1},
    {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1},
    {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1},
    {0, -3, -3, -3,10, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2},
    {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1},
    {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1},
    {0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1},
    {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1},
    {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1},
    {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1},
    {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1},
    {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1},
    {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1},
    {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2},
    {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0},
    {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0},
    {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2},
    {-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1},
    {0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1},
    {-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1},
    {-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1},
    {0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, 4}
};
/* Standard BLAST+ a.a. frequencies */
float aafreq[26] =
{
    0.00000, 0.07805, 0.00000, 0.01925, 0.05364, 0.06295, 0.03856, 0.07377, 0.02199, 0.05142, 0.05744, 0.09019,
    0.02243, 0.04487, 0.05203, 0.04264,    0.05129, 0.07120, 0.05841, 0.06441, 0.01330, 0.00000, 0.03216, 0.00000,
    0.00000, 0.00000
};
/* PSSM arrays */
float fratio[MAXSEQLEN][28], pssm[MAXSEQLEN][28];
/* Dump a rude message to standard error and exit */
void
  fail(char *errstr)
{
    fprintf(stderr, "\\n*** %s\\n\\n", errstr);
    exit(-1);
}
/* Convert AA letter to numeric code (0-22 in 3-letter code order) */
int aanum(int ch)
{
    static const int aacvs[] =
    {
    999, 0, 20, 4, 3, 6, 13, 7, 8, 9, 22, 11, 10, 12, 2,
    22, 14, 5, 1, 15, 16, 22, 19, 17, 22, 18, 21
    };
    return (isalpha(ch) ? aacvs[ch & 31] : 22);
}
/* Scan ahead for file tokens */
void findtoken(char *buf, char *token, FILE *ifp)
{
    for (;;)
    {
    if (fscanf(ifp, "%s", buf) != 1)
        fail("Cannot find token in checkpoint file!");
    if (!token[0] || !strcmp(buf, token))
        break;
    }
}
/* Read hex sequence string */
int readhex(char *seq, FILE *ifp)
{
    int ch, aa, nres=0;
    while ((ch = fgetc(ifp)) != EOF)
    if (ch == '\\'')
        break;
    if (ch == EOF)
    fail("Bad sequence record in checkpoint file!");
    for (;;)
    {
    ch = fgetc(ifp);
    if (ch == '\\'')
        break;
    if (isspace(ch))
        continue;
    if (!isxdigit(ch))
        fail("Bad sequence record in checkpoint file!");
    if (ch >= 'A')
        aa = 16 * (10 + ch - 'A');
    else
        aa = 16 * (ch - '0');
    ch = fgetc(ifp);
    if (!isxdigit(ch))
        fail("Bad sequence record in checkpoint file!");
    if (ch >= 'A')
        aa += 10 + ch - 'A';
    else
        aa += ch - '0';
    if (nres > MAXSEQLEN)
        break;
    seq[nres++] = aa;
    }
    return nres;
}
/* This routine will extract PSSM data from a BLAST+ checkpoint file */
int getpssm(char *dseq, FILE *ifp)
{
    int i, j, len;
    float pssmrow[28], val, base, power;
    char buf[4096];
    findtoken(buf, "", ifp);
    if (strcmp(buf, "PssmWithParameters"))
    fail("Unknown checkpoint file format!");
    findtoken(buf, "numColumns", ifp);
    if (fscanf(ifp, "%d", &len) != 1)
    fail("Unknown checkpoint file format!");
    findtoken(buf, "ncbistdaa", ifp);
    if (len != readhex(dseq, ifp))
    fail("Mismatching sequence length in checkpoint file!");
    findtoken(buf, "freqRatios", ifp);
    findtoken(buf, "", ifp);
    for (i=0; i<len; i++)
    for (j=0; j<28; j++)
    {
        findtoken(buf, "", ifp);
        findtoken(buf, "", ifp);
        if (sscanf(buf, "%f", &val) != 1)
        fail("Unknown checkpoint file format!");
        findtoken(buf, "", ifp);
        if (sscanf(buf, "%f", &base) != 1)
        fail("Unknown checkpoint file format!");
        findtoken(buf, "", ifp);
        if (sscanf(buf, "%f", &power) != 1)
        fail("Unknown checkpoint file format!");
        findtoken(buf, "", ifp);

        fratio[i][j] = val * pow(base, power);
    }
    findtoken(buf, "scores", ifp);
    findtoken(buf, "", ifp);
    for (i=0; i<len; i++)
    for (j=0; j<28; j++)
    {
        findtoken(buf, "", ifp);
        if (sscanf(buf, "%f", &val) != 1)
        fail("Unknown checkpoint file format!");
        pssm[i][j] = val;
    }
    return len;
}
int roundint(double x)
{
    x += (x >= 0.0 ? 0.5 : -0.5);

    return (int)x;
}
int main(int argc, char **argv)
{
    int i, j, seqlen=0, nf;
    char seq[MAXSEQLEN];
    double scale, x, y, sxx, sxy;
    FILE *ifp;
    int use_psipred_format=0;
    if (argc != 2)
    fail("Usage: chkparse chk-file");
    ifp = fopen(argv[1], "r");
    if (!ifp)
    fail("Unable to open checkpoint file!");
    seqlen = getpssm(seq, ifp); // read the sequence from input file and save its length
    if (seqlen < 5 || seqlen >= MAXSEQLEN)
    fail("Sequence length error!");
    /* Estimate original scaling factor by weighted least squares regression */
    for (sxx=sxy=i=0; i<seqlen; i++)
    for (j=0; j<26; j++)
        if (fratio[i][j] > EPSILON && aafreq[j] > EPSILON)
        {
        x = log(fratio[i][j] / aafreq[j]);
        y = pssm[i][j];
        sxx += (y*y) * x * x; /* Weight by y^2 */
        sxy += (y*y) * x * y;
        }
    scale = 100.0 * sxy / sxx;
    if(use_psipred_format)
    {
       printf("%d\\n", seqlen); // print sequence length
       for (i=0; i<seqlen; i++)  // print actual sequence
          putchar(ncbicodes[seq[i]]);
       printf("\\n0\\n0\\n0\\n0\\n0\\n0\\n0\\n0\\n0\\n0\\n0\\n0\\n");
       for (i=0; i<seqlen; i++)
       {
         for (j=0; j<28; j++)
               if (ncbicodes[j] != '*')
               {
             if (fratio[i][j] > EPSILON)
                   printf("%d  ", roundint(scale * log(fratio[i][j] / aafreq[j])));
             else
                      printf("%d  ", 100*aamat[aanum(ncbicodes[seq[i]])][aanum(ncbicodes[j])]);
             }
               else
               printf("-32768  ");
         putchar('\\n');
        }
    }else
    {
      //print header
      printf("        ");
      for (j=0; j<28; j++)
          if (ncbicodes[j] != '*')
              printf("   %c   ",ncbicodes[j]);
      putchar('\\n');

      for (i=0; i<seqlen; i++)
      {
         printf("%5d %c ",i+1,ncbicodes[seq[i]]);
         for (j=0; j<28; j++)
            if (ncbicodes[j] != '*')
            {
             if (fratio[i][j] > EPSILON)
                   printf(" %5.2lf ", roundint(scale * log(fratio[i][j] / aafreq[j]))*0.01 );
             else
                      printf(" %5.2lf ", 1.*aamat[aanum(ncbicodes[seq[i]])][aanum(ncbicodes[j])]);
            }
       putchar('\\n');
       }
    }
    return 0;
}
"""

default_c_compiler = sysconfig.get_config_vars('CC')[0]
if default_c_compiler==None : default_c_compiler='cc'

default_parser_executable='./chkparse'

WEBSERVER_MODE=False

def psiblast_checkpoint(sequence,
                        psi_blast_database_no_last_extension,
                        BLAST_PATH='',
                        sequence_name='to_blast',
                        parser_executable=default_parser_executable,
                        c_compiler=default_c_compiler,
                        num_iterations=3,
                        ncpu=2,
                        str_content_of_chkparse=chkparseC,
                        tmp_fasta_file='',
                        tmp_chk_file='',
                        tmp_wop_file=''):

    database=psi_blast_database_no_last_extension
    if not os.path.isfile(parser_executable) :
        if os.path.isfile('chkparse.c') :
            os.system(c_compiler+' -O chkparse.c -lm -o '+parser_executable)
        elif str_content_of_chkparse!=None :
            out=open('chkparse.c','w')
            out.write(str_content_of_chkparse)
            out.close()
            os.system(c_compiler+' -O chkparse.c -lm -o '+parser_executable)
        else :
            raise IOError('***ERROR*** in psiblast_checkpoint() cant find chkparse (nor chkparse.c) in current folder and neither can find %s (its a compiled c file and its required)\n' % (parser_executable) )
    if BLAST_PATH!='' and not os.path.isfile(BLAST_PATH+'makeblastdb') :    #check if the BLAST_PATH is correct
        raise IOError('***ERROR*** in psiblast_checkpoint() path %s doesnt lead to blast directory where makeblastdb should be located' % (BLAST_PATH) )
    seq_file = file(tmp_fasta_file,'w')
    seq_file.write('> '+sequence_name+'\n'+sequence+'\n')
    seq_file.close()
    #check if the blast database has already been built, if not build it
    if (not ( os.path.isfile(database+'.phr') and os.path.isfile(database+'.pin') and os.path.isfile(database+'.psq')) and (not os.path.isfile(database+'.pal'))  ) : # .pal is for large datasets
        sys.stderr.write('\n********  WARNING  ********\n==> the blast database provided (%s) has not yet been compiled by makeblastdb\n    Running makeblastdb... (this will take a VERY LONG TIME, but needs to be done only once unless the database is deleted/renamed).\n    this may also print some ERROR messages "Error: (1431.1) FASTA-Reader:..." which can safely be ignored\n********           ********\n' % (database))
        sys.stderr.flush()
        try :
            os.system(BLAST_PATH+'makeblastdb -dbtype prot -in '+database)
        except :
            print '***ERROR*** in psiblast_checkpoint() cannot build blast database %s maybe you whish to  set BLAST_PATH to correct directory' % (database)
            raise

    #Query pssm and chk_file
    # os.system(BLAST_PATH+'psiblast -query '+tmp_fasta_file+' -db '+database+' -num_iterations '+str(num_iterations)+'  -inclusion_ethresh 0.001 -out_pssm '+tmp_chk_file+' -out_ascii_pssm '+tmp_fasta_file+' -num_alignments 0 -num_threads '+str(ncpu))

    #Query wop (pssm_ascii) and chkfile
    # os.system(BLAST_PATH + 'psiblast -query ' + tmp_fasta_file + ' -db ' + database + ' -num_iterations ' + str(num_iterations) + '  -inclusion_ethresh 0.001 -out_pssm '+tmp_chk_file+' -out_ascii_pssm ' + tmp_wop_file + ' -num_alignments 0 -num_threads ' + str(ncpu) +' > /dev/null')

    #Query wop from chk file
    # if os.path.exists(tmp_chk_file):
    #     os.system(BLAST_PATH+'psiblast -subject '+tmp_fasta_file+' -in_pssm '+tmp_chk_file+ ' -inclusion_ethresh 0.001 -out_ascii_pssm '+tmp_wop_file+ ' -num_alignments 0 ' +  ' > /dev/null')
    # else:
    #     os.system(BLAST_PATH + 'psiblast -query ' + tmp_fasta_file + ' -db ' + database + ' -num_iterations ' + str(
    #         num_iterations) + '  -inclusion_ethresh 0.001 -out_pssm ' + tmp_chk_file + ' -out_ascii_pssm ' + tmp_wop_file + ' -num_alignments 0 -num_threads ' + str(ncpu)
    #          + ' > /dev/null')
    # if not os.path.exists(tmp_chk_file):
    #     return -1
    # redo=False
    # for ind, l in enumerate(open(tmp_wop_file, 'r')):
    #     if ind < 3:
    #         continue
    #     if len(l.strip().split()) < 30:
    #         redo=True
    #     else:
    #         break
    #
    # if redo:
    #     os.system(BLAST_PATH + 'psiblast -query ' + tmp_fasta_file + ' -db ' + database + ' -num_iterations ' + str(
    #         num_iterations) + '  -inclusion_ethresh 0.001 -out_pssm ' + tmp_chk_file + ' -out_ascii_pssm ' + tmp_wop_file + ' -num_alignments 0 -num_threads ' + str(
    #         ncpu)
    #               + ' > /dev/null')
    #     os.system(
    #         BLAST_PATH + 'psiblast -subject ' + tmp_fasta_file + ' -in_pssm ' + tmp_chk_file + ' -inclusion_ethresh 0.001 -out_ascii_pssm ' + tmp_wop_file + ' -num_alignments 0 ' + ' > /dev/null')

    os.system(BLAST_PATH + 'psiblast -query ' + tmp_fasta_file + ' -db ' + database + ' -num_iterations ' + str(num_iterations) + '  -inclusion_ethresh 0.001 -out_pssm ' + tmp_chk_file + ' -out_ascii_pssm ' + tmp_wop_file + ' -num_alignments 0 -num_threads ' + str(ncpu)+ ' > /dev/null')
    os.system('rm -f '+tmp_fasta_file)

    return 1



##############################################################################
##############################################################################
##############################################################################

use_psiblast_checkpoint = True
temporary_file_directory = ''
psiblast_ncpu = 4


def psiblast_sequence(sequence, sequence_name='myseq',
                      uniref90_psi_blast_database=None,
                      psi_blast_path='',
                      folder_with_wop_files='',
                      folder_with_chk_files=''):
    sequence = str(sequence)
    # print 'SEQ:'+sequence
    if type(sequence) is not str:  # we assume is a Seq object
        sequence_name = sequence.id
        sequence = str(sequence.seq)

    basename = sequence_name.replace('|', '').replace(' ', '_').replace('/', '').replace(':', '_')
    if folder_with_chk_files[-1] != '/': folder_with_chk_files += '/'
    if not os.path.isdir(folder_with_chk_files):
        os.system('mkdir ' + folder_with_chk_files)
    if folder_with_wop_files[-1] != '/': folder_with_wop_files += '/'
    if not os.path.isdir(folder_with_wop_files):
        os.system('mkdir ' + folder_with_wop_files)
    fasta_file = folder_with_chk_files+basename+'.fasta'
    chk_file = folder_with_chk_files+basename+'.chk'
    wop_file = folder_with_wop_files + basename + '.wop'
    return psiblast_checkpoint(sequence,
                        uniref90_psi_blast_database,
        BLAST_PATH=psi_blast_path,
        sequence_name=sequence_name,
        num_iterations=3,
        ncpu=psiblast_ncpu,
        tmp_fasta_file=fasta_file,
        tmp_chk_file=chk_file,
        tmp_wop_file=wop_file)


def readFasta(filename):
    with open(filename) as f:
        for line in f:
            (key, val) = line.split()
            yield(key, val)

def getBlast_(id, seq):
    if seq == '':
        seq = util.downloadSeq(id)
    basename = id.replace('|', '').replace(' ', '_').replace('/', '').replace(':', '_')
    if not os.path.exists(util.wop_dir+basename+'.wop') or not os.path.exists(util.chk_dir+basename+'.chk'):
        return psiblast_sequence(seq,
             sequence_name=id,
             uniref90_psi_blast_database=util.uniref90_psi_blast_database,
             psi_blast_path=util.blast_path,
             folder_with_wop_files=util.wop_dir,
             folder_with_chk_files=util.chk_dir)

def getBlast_individual_(id, seq):
    return psiblast_sequence(seq,
        sequence_name=id,
        uniref90_psi_blast_database=util.uniref90_psi_blast_database,
        psi_blast_path=util.blast_path,
        folder_with_wop_files=util.wop_dir,
        folder_with_chk_files=util.chk_dir)
