package org.scalearn.rbm

class Tuple(val input: Layer, val visible: Layer, val hidden: Layer)

//TODO: perhaps we can create factory in Tuple object
class TupleFactory(input: Layer) {
  def create(visible: Layer, hidden: Layer): Tuple = new Tuple(input, visible, hidden)
}

object TupleFactory {
  def apply(input: Layer) = new TupleFactory(input)
}