using FluentAssertions;
using SharpLeNet.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLeNet.Tests.SharpLeNet.Core.Tests;

public class TensorTests
{
    [Fact]
    public void ShouldGetCorrectShape()
    {
        Tensor tensor = new(new double[]{ 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        tensor.Shape.Should().Equal(new int[] { 2, 3 });
    }

    [Fact]
    public void IndexatorShouldWorkCorrectly()
    {
        Tensor tensor = new(new double[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        double elementAt01 = 2;
        double elementAt02 = 3;
        double elementAt00 = 1;
        double elementAt10 = 4;
        double elementAt11 = 5;
        double elementAt12 = 6;
        tensor[0, 0].Should().Be(elementAt00);
        tensor[0, 1].Should().Be(elementAt01);
        tensor[0, 2].Should().Be(elementAt02);
        tensor[1, 0].Should().Be(elementAt10);
        tensor[1, 1].Should().Be(elementAt11);
        tensor[1, 2].Should().Be(elementAt12);
    }

    [Fact]
    public void ReshapeMethodShouldWorkCorrectly()
    {
        Tensor tensor = new(new double[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        Tensor reshaped = tensor.Reshape(3, 2);
        reshaped.Shape.Should().Equal([3, 2]);
        double elementAt11 = 4;
        reshaped[1, 1].Should().Be(elementAt11);
    }

    [Fact]
    public void TransposeMethodShouldTransposeTensor()
    {
        Tensor tensor = new(new double[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        Tensor transposed = tensor.Transpose();
        transposed.Shape.Should().Equal([3, 2]);
        double elementAt20 = 5;
        transposed[2, 0] = elementAt20;
    }
}
