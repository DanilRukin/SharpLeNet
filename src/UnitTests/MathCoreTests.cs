using FluentAssertions;
using LeNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UnitTests
{
    public partial class MathCoreTests
    {
        [Theory]
        [MemberData(nameof(MaxPoolingMatrixSizeData))]
        public void MaxPoolingShouldReturnMatrxWithCorrectSize(double[,] input, int poolSize, int resultMatrixSize)
        {
            double[,] result = MathCore.MaxPooling(input, poolSize);

            result.Should().NotBeNull();
            result.GetLength(0).Should().Be(resultMatrixSize);
            result.GetLength(1).Should().Be(resultMatrixSize);
        }

        [Theory]
        [MemberData(nameof(MaxPoolingMaxValueData))]
        public void MaxPoolingShouldFindMaxValue(double[,] input, int poolSize, double[,] maxValues)
        {
            double[,] result = MathCore.MaxPooling(input, poolSize);

            result.Should().NotBeNull();
            result.Should().BeEquivalentTo(maxValues);
        }

        [Theory]
        [MemberData(nameof(ExpandWithZerosData))]
        public void ExpandByZeroMethod_ShouldCopySourceMatrixInCenterOfResult(double[,] source, int expandCount, double[,] result)
        {
            double[,] actual = MathCore.Expand(source, expandCount, MathCore.ExpandMethod.WithZeros);

            actual.Should().BeEquivalentTo(result);
        }

        [Theory]
        [MemberData(nameof(ExpandByMirrorMethodData))]
        public void ExpandByMirrorMethod_ShouldFillREsultBorderWithMirrorOfSourceBorder(double[,] source, int expandCount, double[,] result)
        {
            double[,] actual = MathCore.Expand(source, expandCount, MathCore.ExpandMethod.Mirror);

            actual.Should().BeEquivalentTo(result);
        }

        [Theory]
        [MemberData(nameof(ExpandByCopyMethodData))]
        public void ExpandCopyMethod_ShouldFillResultBorderWithCopyOfSourceBorder(double[,] source, int expandCount, double[,] result)
        {
            double[,] actual = MathCore.Expand(source, expandCount, MathCore.ExpandMethod.Copy);

            actual.Should().BeEquivalentTo(result);
        }

        [Theory]
        [MemberData(nameof(ConvolutionReduceData))]
        public void Convolution_ShouldReduceSourceMatrixCorrectlyWithUsingDifferentPaddingAndStride(double[,] source, int padding, int stride, double[,] kernel)
        {
            int kernelHeight = kernel.GetLength(0);
            int kernelWidth = kernel.GetLength(1);
            int sourceHeight = source.GetLength(0);
            int sourceWidth = source.GetLength(1);
            int resultHeight = (sourceHeight - kernelHeight + 2 * padding) / stride + 1;
            int resultWidth = (sourceWidth - kernelWidth + 2 * padding) / stride + 1;

            double[,] result = MathCore.Convolution(source, kernel, stride, padding, MathCore.ExpandMethod.WithZeros);

            result.GetLength(0).Should().Be(resultHeight);
            result.GetLength(1).Should().Be(resultWidth);
        }

        [Theory]
        [MemberData(nameof(ConvolutionSourceKernelPaddingStrideAndResultData))]
        public void Convolution_ShouldGetCorrectResult(double[,] source, int padding, int stride, MathCore.ExpandMethod expandMethod, double[,] kernel, double[,] expectedResult)
        {
            double[,] actualResult = MathCore.Convolution(source, kernel, stride, padding, MathCore.ExpandMethod.WithZeros);
            actualResult.Should().BeEquivalentTo(expectedResult);
        }
    }
}
