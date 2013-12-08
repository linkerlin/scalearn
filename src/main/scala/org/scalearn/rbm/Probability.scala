package org.scalearn.rbm

import scala.util._
import scala.math._

class ProbabilityDistribution(random: Random) {
  def uniform(min: Double, max: Double): Double = random.nextDouble() * (max - min) + min
  def bernoulli(input: Layer): Layer = {
    Layer((for (i <- 0 until input.size) yield if (random.nextFloat() < input(i)) 1.0f else 0.0f).toArray)
  }

}
object ProbabilityDistribution {
  def apply() = new ProbabilityDistribution(new Random())
  def apply(random: Random) = new ProbabilityDistribution(random)

}

object Utilities {

  def mean(input: Layer): Float = {
    var m: Double = input.toArray.foldLeft(0.0) { (R, x) => R + x }
    m / input.size toFloat
  }

  def stddev(input: Layer, mean: Float): Float = {
    val sum: Double = input.toArray.foldLeft(0.0) { (R, x) => R + pow(x - mean, 2) }
    sqrt(sum / (input.size - 1)) toFloat
  }

  def sigmoid(x: Float): Float = 1.0f / (1.0f + exp(-x)) toFloat

  def sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))

}