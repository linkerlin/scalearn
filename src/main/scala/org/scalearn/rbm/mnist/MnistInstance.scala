package org.scalearn.rbm.mnist

import org.scalearn.rbm._

class MnistInstance(val label: String, val data: Array[Int]) {
  def display(msg: String): Unit = {
    print("Instance:" + msg + " label:" + label)
    data.foreach(x => print(x + " "))
    println()
  }
}

object MnistInstance {
  def apply(label: String, data: Array[Int]) = new MnistInstance(label, data)
}
