﻿using LeNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UnitTests
{
    public partial class MathCoreTests
    {
        public static IEnumerable<object[]> MaxPoolingMatrixSizeData =>
            new List<object[]>
            {
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2 },
                        { 1, 2 }
                    },
                    2,
                    1
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 3, 4 },
                        { 1, 2, 3, 4 },
                        { 1, 2, 3, 4 },
                        { 1, 2, 3, 4 }
                    },
                    2,
                    2
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 3, 4 },
                        { 1, 2, 3, 4 },
                        { 1, 2, 3, 4 },
                        { 1, 2, 3, 4 }
                    },
                    4,
                    1
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 3, 4 },
                        { 1, 2, 3, 4 },
                        { 1, 2, 3, 4 },
                        { 1, 2, 3, 4 }
                    },
                    1,
                    4
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                    },
                    2,
                    5
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                    },
                    5,
                    2
                }
            };

        public static IEnumerable<object[]> MaxPoolingMaxValueData =>
            new List<object[]>
            {
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                    },
                    2,
                    new double[,]
                    {
                        { 2, 4, 6, 8, 10 },
                        { 2, 4, 6, 8, 10 },
                        { 2, 4, 6, 8, 10 },
                        { 2, 4, 6, 8, 10 },
                        { 2, 4, 6, 8, 10 },                       
                    },
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                    },
                    5,
                    new double[,]
                    {
                        { 5, 10 },
                        { 5, 10 },
                    },
                }
            };

        public static IEnumerable<object[]> ExpandWithZerosData =>
            new List<object[]>
            {
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    2,
                    new double[,]
                    {
                        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 0, 1, 2, 5, 4, 7, 0, 0 },
                        { 0, 0, 5, 7, 8, 6, 2, 0, 0 },
                        { 0, 0, 3, 4, 6, 7, 1, 0, 0 },
                        { 0, 0, 4, 4, 4, 4, 4, 0, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                    }
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    3,
                    new double[,]
                    {
                        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 0, 0, 1, 2, 5, 4, 7, 0, 0, 0 },
                        { 0, 0, 0, 5, 7, 8, 6, 2, 0, 0, 0 },
                        { 0, 0, 0, 3, 4, 6, 7, 1, 0, 0, 0 },
                        { 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                    }
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    1,
                    new double[,]
                    {
                        { 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 1, 2, 5, 4, 7, 0 },
                        { 0, 5, 7, 8, 6, 2, 0 },
                        { 0, 3, 4, 6, 7, 1, 0 },
                        { 0, 4, 4, 4, 4, 4, 0 },
                        { 0, 0, 0, 0, 0, 0, 0 },
                    }
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    0,
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    }
                },
            };

        public static IEnumerable<object[]> ExpandByMirrorMethodData =>
            new List<object[]>
            {
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    2,
                    new double[,]
                    {
                        { 7, 5, 5, 7, 8, 6, 2, 2, 6 },
                        { 2, 1, 1, 2, 5, 4, 7, 7, 4 },
                        { 2, 1, 1, 2, 5, 4, 7, 7, 4 },
                        { 7, 5, 5, 7, 8, 6, 2, 2, 6 },
                        { 4, 3, 3, 4, 6, 7, 1, 1, 7 },
                        { 4, 4, 4, 4, 4, 4, 4, 4, 4 },
                        { 4, 4, 4, 4, 4, 4, 4, 4, 4 },
                        { 4, 3, 3, 4, 6, 7, 1, 1, 7 },
                    }
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    3,
                    new double[,]
                    {
                        { 6, 4, 3, 3, 4, 6, 7, 1, 1, 7, 6 },
                        { 8, 7, 5, 5, 7, 8, 6, 2, 2, 6, 8 },
                        { 5, 2, 1, 1, 2, 5, 4, 7, 7, 4, 5 },
                        { 5, 2, 1, 1, 2, 5, 4, 7, 7, 4, 5 },
                        { 8, 7, 5, 5, 7, 8, 6, 2, 2, 6, 8 },
                        { 6, 4, 3, 3, 4, 6, 7, 1, 1, 7, 6 },
                        { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 },
                        { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 },
                        { 6, 4, 3, 3, 4, 6, 7, 1, 1, 7, 6 },
                        { 8, 7, 5, 5, 7, 8, 6, 2, 2, 6, 8 },
                    }
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    1,
                    new double[,]
                    {
                        { 1, 1, 2, 5, 4, 7, 7 },
                        { 1, 1, 2, 5, 4, 7, 7 },
                        { 5, 5, 7, 8, 6, 2, 2 },
                        { 3, 3, 4, 6, 7, 1, 1 },
                        { 4, 4, 4, 4, 4, 4, 4 },
                        { 4, 4, 4, 4, 4, 4, 4 },
                    }
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    0,
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    }
                },
            };

        public static IEnumerable<object[]> ExpandByCopyMethodData =>
            new List<object[]>
            {
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    2,
                    new double[,]
                    {
                        { 1, 2, 1, 2, 5, 4, 7, 4, 7 },
                        { 5, 7, 5, 7, 8, 6, 2, 6, 2 },
                        { 1, 2, 1, 2, 5, 4, 7, 4, 7 },
                        { 5, 7, 5, 7, 8, 6, 2, 6, 2 },
                        { 3, 4, 3, 4, 6, 7, 1, 7, 1 },
                        { 4, 4, 4, 4, 4, 4, 4, 4, 4 },
                        { 3, 4, 3, 4, 6, 7, 1, 7, 1 },
                        { 4, 4, 4, 4, 4, 4, 4, 4, 4 },
                    }
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    3,
                    new double[,]
                    {
                        { 1, 2, 5, 1, 2, 5, 4, 7, 5, 4, 7 },
                        { 5, 7, 8, 5, 7, 8, 6, 2, 8, 6, 2 },
                        { 3, 4, 6, 3, 4, 6, 7, 1, 6, 7, 1 },
                        { 1, 2, 5, 1, 2, 5, 4, 7, 5, 4, 7 },
                        { 5, 7, 8, 5, 7, 8, 6, 2, 8, 6, 2 },
                        { 3, 4, 6, 3, 4, 6, 7, 1, 6, 7, 1 },
                        { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 },
                        { 5, 7, 8, 5, 7, 8, 6, 2, 8, 6, 2 },
                        { 3, 4, 6, 3, 4, 6, 7, 1, 6, 7, 1 },
                        { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 },
                    }
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    1,
                    new double[,]
                    {
                        { 1, 1, 2, 5, 4, 7, 7 },
                        { 1, 1, 2, 5, 4, 7, 7 },
                        { 5, 5, 7, 8, 6, 2, 2 },
                        { 3, 3, 4, 6, 7, 1, 1 },
                        { 4, 4, 4, 4, 4, 4, 4 },
                        { 4, 4, 4, 4, 4, 4, 4 },
                    }
                },
                new object[]
                {
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    },
                    0,
                    new double[,]
                    {
                        { 1, 2, 5, 4, 7 },
                        { 5, 7, 8, 6, 2 },
                        { 3, 4, 6, 7, 1 },
                        { 4, 4, 4, 4, 4 }
                    }
                },
            };


        public static IEnumerable<object[]> ConvolutionReduceData =>
            new List<object[]>
            {
                new object[]
                {
                    // source
                    CreateFilledMatrix(9, 9, 5),
                    // padding
                    0,
                    // stride
                    1,
                    // kernel
                    CreateFilledMatrix(3, 3, 1)
                },

                new object[]
                {
                    // source
                    CreateFilledMatrix(10, 10, 5),
                    // padding
                    2,
                    // stride
                    4,
                    // kernel
                    CreateFilledMatrix(3, 3, 1)
                },

                new object[]
                {
                    // source
                    CreateFilledMatrix(8, 8, 5),
                    // padding
                    1,
                    // stride
                    1,
                    // kernel
                    CreateFilledMatrix(3, 3, 1)
                },
            };


        private static double[,] CreateFilledMatrix(int rows, int cols, double initialValue)
        {
            double[,] result = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int  j = 0; j < cols; j++)
                {
                    result[i, j] = initialValue;
                }
            }
            return result;
        }

        public static IEnumerable<object[]> ConvolutionSourceKernelPaddingStrideAndResultData =>
            new List<object[]>
            {
                new object[]
                {
                    // source
                    new double[,]
                    {
                        { 3, 3, 2, 1, 0 },
                        { 0, 0, 1, 3, 1 },
                        { 3, 1, 2, 2, 3 },
                        { 2, 0, 0, 2, 2 },
                        { 2, 0, 0, 0, 1 }
                    },
                    0, 1, MathCore.ExpandMethod.WithZeros, // padding, stride, ExpandMethod
                    // kernel
                    new double[,]
                    {
                        { 0, 1, 2 },
                        { 2, 2, 0 },
                        { 0, 1, 2 }
                    },
                    // result
                    new double[,]
                    {
                        { 12, 12, 17 },
                        { 10, 17, 19 },
                        { 9, 6, 14 }
                    }
                },

                new object[]
                {
                    // source
                    new double[,]
                    {
                        { 3, 3, 2, 1, 0 },
                        { 0, 0, 1, 3, 1 },
                        { 3, 1, 2, 2, 3 },
                        { 2, 0, 0, 2, 2 },
                        { 2, 0, 0, 0, 1 }
                    },
                    1, 2, MathCore.ExpandMethod.WithZeros, // padding, stride, ExpandMethod
                    // kernel
                    new double[,]
                    {
                        { 0, 1, 2 },
                        { 2, 2, 0 },
                        { 0, 1, 2 }
                    },
                    // result
                    new double[,]
                    {
                        { 6, 17, 3 },
                        { 8, 17, 13 },
                        { 6, 4, 4 }
                    }
                }
            };

    }
}
