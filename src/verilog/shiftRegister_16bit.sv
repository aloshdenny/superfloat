module ShiftRegister #(parameter SIZE = 16, DEPTH = 8) (
    input wire clk,
    input wire reset,
    input wire shift_en,
    input wire [SIZE-1:0] data_in,
    output reg [SIZE-1:0] data_out
);
    
    reg [SIZE-1:0] shift_reg [DEPTH-1:0]; // Single chain of 8 registers, each 16-bit
    integer i;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                shift_reg[i] <= 0;
            end
            data_out <= 0;
        end else if (shift_en) begin
            for (i = DEPTH-1; i > 0; i = i - 1) begin
                shift_reg[i] <= shift_reg[i-1];
            end
            shift_reg[0] <= data_in;
            data_out <= shift_reg[DEPTH-1];
        end
    end
endmodule
